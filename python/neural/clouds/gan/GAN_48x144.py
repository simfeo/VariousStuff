"""DCGAN at the sprites' own aspect ratio.

A centred cloud covers about 2% of a 256x256 canvas, so both networks there mostly model
empty background. In a 48x144 frame the same cloud fills roughly a third. Padding happens
in the dataset, so this reads the native sprites in ../images with no prepared copy.
"""
import argparse
import copy
import datetime
import os
import random
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from GAN_256_4_chanels import CHANNELS, LR, REAL_LABEL, UpBlock, save_rgba, weights_init

IMG_H, IMG_W = 48, 144
# Four doublings from here land exactly on 48x144, so no cropping or odd padding is
# needed anywhere in the generator.
BASE_H, BASE_W = 3, 9
LATENT_DIM = 128
BATCH_SIZE = 32
# Maximum offset in pixels, applied independently on each axis.
SHIFT = 2
EMA_DECAY = 0.999
# Until this many steps the average is just a copy: at 0.999 the random initialization
# still weighs 62 percent after 480 steps, and early samples show nothing.
EMA_WARMUP_STEPS = 500
SAMPLE_EVERY = 10
CHECKPOINT_EVERY = 50
DATA_DIR = "../images"
OUT_DIR = "out_gan_48"


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 512 * BASE_H * BASE_W)
        self.bn0 = nn.BatchNorm2d(512)
        self.ups = nn.Sequential(
            UpBlock(512, 256),   # 6x18
            UpBlock(256, 128),   # 12x36
            UpBlock(128, 64),    # 24x72
            UpBlock(64, 32),     # 48x144
        )
        self.out = nn.Conv2d(32, CHANNELS, 3, padding=1)

    def forward(self, z):
        x = self.lin1(z).view(-1, 512, BASE_H, BASE_W)
        x = torch.relu(self.bn0(x))
        return torch.tanh(self.out(self.ups(x)))


class MinibatchStdDev(nn.Module):
    """Appends the batch's own feature spread as an extra channel.

    Collapse is invisible to a discriminator that judges each sample alone: eight
    identical clouds each look as plausible as one. This makes sameness a visible feature."""

    def forward(self, x):
        std = x.std(0, unbiased=False).mean()
        return torch.cat([x, std.expand(x.size(0), 1, x.size(2), x.size(3))], 1)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(CHANNELS, 64, norm=False),   # 24x72
            *block(64, 128),                    # 12x36
            *block(128, 256),                   # 6x18
            *block(256, 512),                   # 3x9
            MinibatchStdDev(),
            nn.Conv2d(512 + 1, 1, (BASE_H, BASE_W)),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


class SpriteDataset(Dataset):
    """Sprites placed on a 48x144 canvas, offset by up to SHIFT pixels each way.

    Without it the discriminator sees the same 523 tensors every epoch and memorizes them.
    Composited per item rather than pre-expanded, which would cost a gigabyte."""

    def __init__(self, root_dir, shift=SHIFT):
        self.shift = shift
        names = sorted(n for n in os.listdir(root_dir) if n.lower().endswith(".png"))
        self.sprites = []
        for name in names:
            image = iio.imread(os.path.join(root_dir, name))
            h, w = image.shape[:2]
            if h > IMG_H or w > IMG_W:
                scale = min(IMG_H / h, IMG_W / w)
                new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                ys = (np.arange(new_h) / scale).astype(int).clip(0, h - 1)
                xs = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
                image = image[ys][:, xs]

            opaque = image[..., 3] >= 128
            image = image.copy()
            image[~opaque] = 0
            image[..., 3] = np.where(opaque, 255, 0)
            self.sprites.append(image)

    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, idx):
        sprite = self.sprites[idx]
        h, w = sprite.shape[:2]
        top, left = (IMG_H - h) // 2, (IMG_W - w) // 2
        # 51 of 523 sprites fill the frame to within two rows, so the offset is clamped
        # per sprite rather than globally.
        dy = random.randint(-min(self.shift, top), min(self.shift, IMG_H - h - top))
        dx = random.randint(-min(self.shift, left), min(self.shift, IMG_W - w - left))

        canvas = np.zeros((IMG_H, IMG_W, 4), np.float32)
        canvas[top + dy:top + dy + h, left + dx:left + dx + w] = sprite
        tensor = torch.from_numpy(canvas / 255.0).permute(2, 0, 1)
        return tensor * 2 - 1  # to [-1, 1], matching the generator's tanh


class GAN(L.LightningModule):
    def __init__(self, latent_dim=LATENT_DIM, lr=LR, data_dir=DATA_DIR):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.criterion = nn.BCEWithLogitsLoss()
        # Sampled from instead of the live generator. Adversarial training oscillates, and
        # the best epoch of the previous run was 550 out of 600 by luck rather than trend.
        self.gen_ema = copy.deepcopy(self.generator).requires_grad_(False)
        self.register_buffer("validation_z", torch.randn(8, latent_dim))
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        n = batch.size(0)
        opt_g, opt_d = self.optimizers()

        z = torch.randn(n, self.hparams.latent_dim, device=self.device)
        fake = self(z).detach()
        d_loss = 0.5 * (
            self.criterion(self.discriminator(batch),
                           torch.full((n, 1), REAL_LABEL, device=self.device))
            + self.criterion(self.discriminator(fake), torch.zeros(n, 1, device=self.device)))
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()

        z = torch.randn(n, self.hparams.latent_dim, device=self.device)
        g_loss = self.criterion(self.discriminator(self(z)),
                                torch.ones(n, 1, device=self.device))
        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()
        self.update_ema()

        self.log_dict({"d_loss": d_loss, "g_loss": g_loss}, prog_bar=True)

    @torch.no_grad()
    def update_ema(self):
        decay = EMA_DECAY if self.global_step >= EMA_WARMUP_STEPS else 0.0
        for ema, live in zip(self.gen_ema.parameters(), self.generator.parameters()):
            ema.lerp_(live, 1 - decay)
        # Buffers are BatchNorm running statistics, which are not gradient-tracked and
        # have to be carried over rather than averaged.
        for ema, live in zip(self.gen_ema.buffers(), self.generator.buffers()):
            ema.copy_(live)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        return DataLoader(SpriteDataset(self.hparams.data_dir), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, persistent_workers=True,
                          drop_last=True)

    def on_train_epoch_end(self):
        epoch = self.current_epoch + 1
        metrics = self.trainer.callback_metrics
        print("epoch %4d  d_loss %.4f  g_loss %.4f"
              % (epoch, metrics["d_loss"].item(), metrics["g_loss"].item()), flush=True)
        if epoch % SAMPLE_EVERY and epoch != 1:
            return
        with torch.no_grad():
            sample = self.gen_ema.eval()(self.validation_z)
        grid = torchvision.utils.make_grid(sample, nrow=2, padding=4,
                                           normalize=True, value_range=(-1, 1))
        os.makedirs(OUT_DIR, exist_ok=True)
        torchvision.utils.save_image(grid, os.path.join(OUT_DIR, "epoch_%04d.png" % epoch))


def train(args):
    model = GAN(data_dir=args.data_dir)
    # Kept, not rolled over: the best samples do not come from the last epoch.
    # Lightning expands "{epoch:04d}" to "epoch=0049", so the template adds no prefix.
    keeper = ModelCheckpoint(dirpath=args.ckpt_dir, filename="{epoch:04d}",
                             every_n_epochs=CHECKPOINT_EVERY, save_top_k=-1,
                             save_on_train_epoch_end=True)
    trainer = L.Trainer(max_epochs=args.epochs, log_every_n_steps=10,
                        enable_progress_bar=False, callbacks=[keeper])
    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(args.ckpt_dir, "final.ckpt"))
    print("saved %s" % os.path.join(args.ckpt_dir, "final.ckpt"), flush=True)


def generate(args):
    model = GAN.load_from_checkpoint(args.checkpoint)
    model.eval().to(args.device)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with torch.no_grad():
        z = torch.randn(args.count, model.hparams.latent_dim, device=args.device)
        images = model.gen_ema(z)
    for i, image in enumerate(images):
        save_rgba(image, Path(args.out) / ("fake-%s-%02d.png" % (stamp, i)))
    print("wrote %d images to %s" % (args.count, args.out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--ckpt-dir", default="ckpt_48")
    parser.add_argument("--checkpoint", default="ckpt_48/final.ckpt")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    generate(args) if args.generate else train(args)
