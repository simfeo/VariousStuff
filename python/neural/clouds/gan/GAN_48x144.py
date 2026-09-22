"""DCGAN at the sprites' own aspect ratio.

The 256x256 version works but spends almost everything on emptiness: a centred cloud
covers about 2% of that canvas, so both networks mostly model transparent background.
A 48x144 frame is the shape of the sprites themselves, and the same cloud fills roughly
a third of it.

Padding happens in the dataset, so this reads the native sprites in ../images directly
and needs no prepared copy on disk.
"""
import argparse
import datetime
import os
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
            nn.Conv2d(512, 1, (BASE_H, BASE_W)),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


class SpriteDataset(Dataset):
    """Native sprites centred on a 48x144 canvas, held in memory as tensors.

    523 sprites at this size are a few megabytes, so they are padded once at startup
    rather than on every epoch."""

    def __init__(self, root_dir):
        names = sorted(n for n in os.listdir(root_dir) if n.lower().endswith(".png"))
        self.items = []
        for name in names:
            image = iio.imread(os.path.join(root_dir, name))
            h, w = image.shape[:2]
            if h > IMG_H or w > IMG_W:
                scale = min(IMG_H / h, IMG_W / w)
                new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                ys = (np.arange(new_h) / scale).astype(int).clip(0, h - 1)
                xs = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
                image = image[ys][:, xs]
                h, w = image.shape[:2]

            canvas = np.zeros((IMG_H, IMG_W, 4), np.float32)
            top, left = (IMG_H - h) // 2, (IMG_W - w) // 2
            canvas[top:top + h, left:left + w] = image
            opaque = canvas[..., 3] >= 128
            canvas[~opaque] = 0
            canvas[..., 3] = np.where(opaque, 255, 0)
            tensor = torch.from_numpy(canvas / 255.0).permute(2, 0, 1)
            self.items.append(tensor * 2 - 1)  # to [-1, 1], matching the generator's tanh

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class GAN(L.LightningModule):
    def __init__(self, latent_dim=LATENT_DIM, lr=LR, data_dir=DATA_DIR):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.criterion = nn.BCEWithLogitsLoss()
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

        self.log_dict({"d_loss": d_loss, "g_loss": g_loss}, prog_bar=True)

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
        self.generator.eval()
        with torch.no_grad():
            sample = self(self.validation_z)
        self.generator.train()
        grid = torchvision.utils.make_grid(sample, nrow=2, padding=4,
                                           normalize=True, value_range=(-1, 1))
        os.makedirs(OUT_DIR, exist_ok=True)
        torchvision.utils.save_image(grid, os.path.join(OUT_DIR, "epoch_%04d.png" % epoch))


def train(args):
    model = GAN(data_dir=args.data_dir)
    # Every 50 epochs, kept, not rolled over: the 256 run peaked around epoch 200 and
    # degraded afterwards, and with only a final checkpoint there was nothing to go back to.
    # Lightning expands "{epoch:04d}" to "epoch=0049", so the template carries no prefix
    # of its own: writing "epoch{epoch:04d}" produced files named "epochepoch=0049.ckpt".
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
        images = model(z)
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
