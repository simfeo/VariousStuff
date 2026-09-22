"""DCGAN that classifies palette indices instead of regressing RGBA.

Measured against the dataset's 552 colours, the regression version puts 0.03% of its
pixels on a palette colour and never once repeats a colour in two neighbouring pixels,
where the sources do so 45% of the time. It draws a dithered gradient shaped like a
cloud. Here the generator picks one of 256 palette entries or the transparent class per
pixel, so both properties come from the output layer rather than from training longer.

argmax has no gradient, so the discriminator is fed a straight-through Gumbel-Softmax
sample: hard one-hot forward, soft gradient backward.
"""
import argparse
import collections
import copy
import datetime
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from GAN_256_4_chanels import LR, REAL_LABEL, UpBlock, weights_init
from GAN_48x144 import (BASE_H, BASE_W, EMA_DECAY, EMA_WARMUP_STEPS, IMG_H, IMG_W,
                        MinibatchStdDev, SHIFT)

PALETTE_SIZE = 256
TRANSPARENT = PALETTE_SIZE
N_CLASSES = PALETTE_SIZE + 1
LATENT_DIM = 128
BATCH_SIZE = 32
SAMPLE_EVERY = 10
CHECKPOINT_EVERY = 50
# Annealed over the run. Starting at 2.0 kept the sample near uniform for the first
# hundred epochs, which the discriminator separated from real sprites without ever
# looking at shape: its loss sat at 0.1635 for fifty epochs and the generator never moved.
TAU_START, TAU_END = 1.0, 0.3
# Both sides pass through the same relaxation, soft rather than hard, for this many
# epochs. Softening only the fake would hand the discriminator "blended colour means
# fake" as a free answer, so real sprites are relaxed identically.
#
# Set beyond the epoch count, which means never switching to hard. Switching at 200 cost
# everything the soft phase had built: the discriminator separated hard samples instantly
# (d_loss 0.67 to 0.187 in one epoch) and a hundred epochs later the generator had
# collapsed to one sprite regardless of z. Sampling takes argmax either way, so the saved
# output is discrete whether or not the discriminator ever judged a hard sample.
SOFT_EPOCHS = 10 ** 9
# One-hot scaled to logits before relaxing. Gumbel noise has a standard deviation near
# 1.28, so at this scale it practically never flips the true class.
REAL_LOGIT_SCALE = 20.0
# The discriminator wins this matchup easily; it gets a quarter of the generator's rate
# and half as many steps.
D_LR_FACTOR = 0.25
G_STEPS = 2
DATA_DIR = "../images"
OUT_DIR = "out_gan_pal"


def build_palette(img_dir, names):
    counts = collections.Counter()
    for name in names:
        image = iio.imread(os.path.join(img_dir, name))
        counts.update(map(tuple, image[image[..., 3] > 128][:, :3]))
    return np.array([c for c, _ in counts.most_common(PALETTE_SIZE)], dtype=np.int32)


def quantize(image, palette):
    """Every pixel to a palette index, or to the transparent class. Colours outside the
    top 256 snap to the nearest entry rather than being dropped."""
    rgb = image[..., :3].astype(np.int32)  # int32: squared differences overflow int16
    idx = ((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(-1).argmin(-1)
    idx = idx.astype(np.int64)
    idx[image[..., 3] <= 128] = TRANSPARENT
    return idx


class IndexSpriteDataset(Dataset):
    """Class maps on a 48x144 canvas, offset by up to SHIFT pixels each way.

    Returns the sprite's own index alongside the map, which the pretraining stage needs
    to look up that sprite's latent code."""

    def __init__(self, root_dir, shift=SHIFT):
        self.shift = shift
        names = sorted(n for n in os.listdir(root_dir) if n.lower().endswith(".png"))
        self.palette = build_palette(root_dir, names)
        self.sprites = []
        for name in names:
            image = iio.imread(os.path.join(root_dir, name))
            h, w = image.shape[:2]
            if h > IMG_H or w > IMG_W:
                scale = min(IMG_H / h, IMG_W / w)
                ys = (np.arange(max(1, int(h * scale))) / scale).astype(int).clip(0, h - 1)
                xs = (np.arange(max(1, int(w * scale))) / scale).astype(int).clip(0, w - 1)
                image = image[ys][:, xs]
            self.sprites.append(quantize(image, self.palette))

    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, idx):
        sprite = self.sprites[idx]
        h, w = sprite.shape
        top, left = (IMG_H - h) // 2, (IMG_W - w) // 2
        dy = np.random.randint(-min(self.shift, top), min(self.shift, IMG_H - h - top) + 1)
        dx = np.random.randint(-min(self.shift, left), min(self.shift, IMG_W - w - left) + 1)

        canvas = np.full((IMG_H, IMG_W), TRANSPARENT, dtype=np.int64)
        canvas[top + dy:top + dy + h, left + dx:left + dx + w] = sprite
        return torch.from_numpy(canvas), idx


def to_rgba(onehot, palette_t):
    """One-hot class planes to the RGBA tensor the discriminator reads.

    Differentiable in onehot, so gradients from the discriminator reach the generator's
    logits through the straight-through sample."""
    rgb = torch.einsum("nchw,cd->ndhw", onehot[:, :PALETTE_SIZE], palette_t)
    alpha = 1 - onehot[:, TRANSPARENT:TRANSPARENT + 1]
    return torch.cat([rgb * alpha, alpha], 1) * 2 - 1


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 512 * BASE_H * BASE_W)
        self.bn0 = nn.BatchNorm2d(512)
        self.ups = nn.Sequential(
            UpBlock(512, 256), UpBlock(256, 128), UpBlock(128, 64), UpBlock(64, 32))
        self.out = nn.Conv2d(32, N_CLASSES, 1)

    def forward(self, z):
        x = torch.relu(self.bn0(self.lin1(z).view(-1, 512, BASE_H, BASE_W)))
        return self.out(self.ups(x))  # logits, one per class per pixel


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
            *block(4, 64, norm=False), *block(64, 128), *block(128, 256), *block(256, 512),
            MinibatchStdDev(), nn.Conv2d(512 + 1, 1, (BASE_H, BASE_W)))

    def forward(self, x):
        return self.net(x).view(-1, 1)


class PaletteGAN(L.LightningModule):
    def __init__(self, palette, latent_dim=LATENT_DIM, lr=LR, data_dir=DATA_DIR,
                 max_epochs=600):
        super().__init__()
        self.save_hyperparameters(ignore=["palette"])
        self.register_buffer("palette_t", palette)
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.criterion = nn.BCEWithLogitsLoss()
        self.gen_ema = copy.deepcopy(self.generator).requires_grad_(False)
        self.register_buffer("validation_z", torch.randn(8, latent_dim))
        self.automatic_optimization = False

    @property
    def tau(self):
        progress = self.current_epoch / max(1, self.hparams.max_epochs - 1)
        return TAU_START + (TAU_END - TAU_START) * progress

    @property
    def hard(self):
        return self.current_epoch >= SOFT_EPOCHS

    def relax(self, logits):
        """The one operator both sides go through, so neither can be told from the other
        by how sharp its class distribution is."""
        return F.gumbel_softmax(logits, tau=self.tau, hard=self.hard, dim=1)

    def sample_rgba(self, z):
        return to_rgba(self.relax(self.generator(z)), self.palette_t)

    def real_rgba(self, indices):
        logits = F.one_hot(indices, N_CLASSES).permute(0, 3, 1, 2).float() * REAL_LOGIT_SCALE
        return to_rgba(self.relax(logits), self.palette_t)

    def training_step(self, batch, batch_idx):
        batch = batch[0]
        n = batch.size(0)
        opt_g, opt_d = self.optimizers()
        real = self.real_rgba(batch)

        z = torch.randn(n, self.hparams.latent_dim, device=self.device)
        fake = self.sample_rgba(z).detach()
        d_loss = 0.5 * (
            self.criterion(self.discriminator(real),
                           torch.full((n, 1), REAL_LABEL, device=self.device))
            + self.criterion(self.discriminator(fake), torch.zeros(n, 1, device=self.device)))
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()

        for _ in range(G_STEPS):
            z = torch.randn(n, self.hparams.latent_dim, device=self.device)
            g_loss = self.criterion(self.discriminator(self.sample_rgba(z)),
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
        for ema, live in zip(self.gen_ema.buffers(), self.generator.buffers()):
            ema.copy_(live)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr * D_LR_FACTOR,
                                 betas=(0.5, 0.999))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        return DataLoader(IndexSpriteDataset(self.hparams.data_dir), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, persistent_workers=True,
                          drop_last=True)

    @torch.no_grad()
    def generate_indices(self, z):
        return self.gen_ema.eval()(z).argmax(1)

    def on_train_epoch_end(self):
        epoch = self.current_epoch + 1
        m = self.trainer.callback_metrics
        print("epoch %4d  d_loss %.4f  g_loss %.4f  tau %.2f  %s"
              % (epoch, m["d_loss"].item(), m["g_loss"].item(), self.tau,
                 "hard" if self.hard else "soft"), flush=True)
        if epoch % SAMPLE_EVERY and epoch != 1:
            return
        os.makedirs(OUT_DIR, exist_ok=True)
        indices = self.generate_indices(self.validation_z)
        save_sheet(indices, self.palette_t, os.path.join(OUT_DIR, "epoch_%04d.png" % epoch))


def indices_to_rgba(indices, palette_t):
    idx = indices.cpu().numpy()
    palette = (palette_t.cpu().numpy() * 255).astype(np.uint8)
    out = np.zeros((*idx.shape, 4), np.uint8)
    opaque = idx != TRANSPARENT
    out[opaque, :3] = palette[idx[opaque]]
    out[opaque, 3] = 255
    return out


def save_sheet(indices, palette_t, path, scale=3, pad=4, cols=2):
    """Preview composited over grey, not normalized.

    make_grid(normalize=True) rescales the alpha channel along with the colours, which
    renders sub-threshold haze as opaque background and hides the real silhouette."""
    tiles = indices_to_rgba(indices, palette_t)
    n, h, w = tiles.shape[0], tiles.shape[1] * scale, tiles.shape[2] * scale
    rows = (n + cols - 1) // cols
    sheet = np.zeros((rows * (h + pad) + pad, cols * (w + pad) + pad, 4), np.uint8)
    sheet[..., :3] = 90
    sheet[..., 3] = 255
    for k in range(n):
        tile = tiles[k].repeat(scale, 0).repeat(scale, 1)
        y, x = pad + (k // cols) * (h + pad), pad + (k % cols) * (w + pad)
        cell = sheet[y:y + h, x:x + w]
        alpha = tile[..., 3:4] / 255.0
        cell[..., :3] = (tile[..., :3] * alpha + cell[..., :3] * (1 - alpha)).astype(np.uint8)
    iio.imwrite(path, sheet)


def load_palette(data_dir):
    names = sorted(n for n in os.listdir(data_dir) if n.lower().endswith(".png"))
    return torch.from_numpy(build_palette(data_dir, names).astype(np.float32) / 255.0)


def pretrain_generator(model, data_dir, epochs, device, out_dir):
    """Fit the generator to the sprites by cross entropy before any discriminator exists.

    Three adversarial runs from random initialization never found a stable path: the
    generator either sat in noise with the discriminator's loss pinned at 0.163, or
    collapsed to an empty frame. Cross entropy has a gradient towards the data at every
    step, which an adversarial loss does not, so it is used to reach cloud-shaped output
    first and the discriminator only refines it afterwards.

    There is no encoder to produce a latent for a given sprite, so each sprite gets its
    own learned code, optimized alongside the generator. Codes are projected onto the
    sphere of radius sqrt(latent_dim), where standard normal samples also concentrate, so
    the adversarial stage can keep drawing z from a normal distribution.
    """
    dataset = IndexSpriteDataset(data_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dim = model.hparams.latent_dim
    codes = nn.Embedding(len(dataset), dim).to(device)
    nn.init.normal_(codes.weight)
    generator = model.generator.to(device).train()
    opt = torch.optim.Adam(list(generator.parameters()) + list(codes.parameters()), lr=1e-3)
    radius = dim ** 0.5

    for epoch in range(1, epochs + 1):
        total, batches = 0.0, 0
        for maps, idx in loader:
            maps, idx = maps.to(device), idx.to(device)
            loss = F.cross_entropy(generator(codes(idx)), maps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                w = codes.weight
                w.mul_(radius / w.norm(dim=1, keepdim=True))
            total += loss.item()
            batches += 1
        print("pretrain %4d  ce %.4f" % (epoch, total / batches), flush=True)
        if epoch % SAMPLE_EVERY == 0 or epoch == 1:
            os.makedirs(out_dir, exist_ok=True)
            with torch.no_grad():
                z = torch.randn(8, dim, device=device)
                indices = generator(z * radius / z.norm(dim=1, keepdim=True)).argmax(1)
            save_sheet(indices, model.palette_t.to(device),
                       os.path.join(out_dir, "pretrain_%04d.png" % epoch))

    model.gen_ema.load_state_dict(generator.state_dict())
    print("pretraining done", flush=True)


def train(args):
    model = PaletteGAN(load_palette(args.data_dir), data_dir=args.data_dir,
                       max_epochs=args.epochs)
    if args.pretrain_epochs:
        pretrain_generator(model, args.data_dir, args.pretrain_epochs, args.device, args.out)
    keeper = ModelCheckpoint(dirpath=args.ckpt_dir, filename="{epoch:04d}",
                             every_n_epochs=CHECKPOINT_EVERY, save_top_k=-1,
                             save_on_train_epoch_end=True)
    trainer = L.Trainer(max_epochs=args.epochs, log_every_n_steps=10,
                        enable_progress_bar=False, callbacks=[keeper])
    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(args.ckpt_dir, "final.ckpt"))
    print("saved %s" % os.path.join(args.ckpt_dir, "final.ckpt"), flush=True)


def generate(args):
    model = PaletteGAN.load_from_checkpoint(args.checkpoint, palette=load_palette(args.data_dir))
    model.eval().to(args.device)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    z = torch.randn(args.count, model.hparams.latent_dim, device=args.device)
    rgba = indices_to_rgba(model.generate_indices(z), model.palette_t)
    for i, image in enumerate(rgba):
        iio.imwrite(Path(args.out) / ("fake-%s-%02d.png" % (stamp, i)), image)
    print("wrote %d images to %s" % (args.count, args.out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--pretrain-epochs", type=int, default=150,
                        help="cross entropy epochs before the discriminator; 0 to skip")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--ckpt-dir", default="ckpt_pal")
    parser.add_argument("--checkpoint", default="ckpt_pal/final.ckpt")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    generate(args) if args.generate else train(args)
