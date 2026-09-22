import collections
import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

DATA_DIR = "../images"
IMG_H, IMG_W = 48, 144
CODE_H, CODE_W = 12, 36
PALETTE_SIZE = 256
EMBED_DIM = 64
N_CODES = 512
COMMIT_COST = 0.25
EMA_DECAY = 0.99
EMA_EPS = 1e-5
DEAD_CODE_THRESHOLD = 1.0
EPOCHS = 600
# With the dilated blocks the loss is flat from 400 to 600 (0.0529 to 0.0527) and only
# drops again by 800, which tracked a jump in copying of the training sprites from 6.4
# to 14.7 percent agreement. Stop before that late descent.
PRIOR_EPOCHS = 400
PRIOR_HIDDEN = 128
PRIOR_BLOCKS = 12
# Undilated blocks add one cell of context each, so 12 of them reached only ~14 of the
# 36 columns and the model kept starting a second cloud it could not see. Cycling the
# dilation covers the full width at the same depth and parameter count: each cycle
# spans 1+2+4+8 = 15 columns, three cycles plus the entry convolution reach 47.
PRIOR_DILATIONS = (1, 2, 4, 8)
PRIOR_DROPOUT = 0.25
TEMPERATURE = 0.8
TOP_K = 35

# Reuse an already trained VQ-VAE instead of retraining it, so prior settings can be
# iterated on their own. Set to None to train the whole thing from scratch.
VQVAE_CHECKPOINT = "lightning_logs/version_57/checkpoints/epoch=599-step=19800.ckpt"
BATCH_SIZE = 16
OUT_DIR = "out_pixel"
N_SAMPLES = 16

# Class PALETTE_SIZE means "transparent". Folding alpha into the same classification
# makes the silhouette discrete by construction, so no alpha thresholding is needed.
TRANSPARENT = PALETTE_SIZE


def build_palette(img_dir, names):
    counts = collections.Counter()
    for name in names:
        image = iio.imread(os.path.join(img_dir, name))
        opaque = image[image[..., 3] > 128][:, :3]
        counts.update(map(tuple, opaque))
    top = [c for c, _ in counts.most_common(PALETTE_SIZE)]
    return np.array(top, dtype=np.int32)


def quantize(image, palette):
    """Map every pixel to a palette index, or to the transparent class. Rare colours
    outside the top 256 snap to the nearest palette entry rather than being dropped."""
    # int32: squared channel differences reach 255^2 and silently overflow int16.
    rgb = image[..., :3].astype(np.int32)
    dist = ((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(-1)
    idx = dist.argmin(-1).astype(np.int64)
    idx[image[..., 3] <= 128] = TRANSPARENT
    return torch.from_numpy(idx)


def fit_canvas(indices):
    canvas = torch.full((IMG_H, IMG_W), TRANSPARENT, dtype=torch.long)
    h, w = indices.shape
    top, left = (IMG_H - h) // 2, (IMG_W - w) // 2
    canvas[top:top + h, left:left + w] = indices
    return canvas


class CloudDataset(Dataset):
    def __init__(self, img_dir):
        names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(".png"))
        self.palette = build_palette(img_dir, names)
        self.maps = []
        for name in names:
            image = iio.imread(os.path.join(img_dir, name))
            if image.shape[0] > IMG_H or image.shape[1] > IMG_W:
                scale = min(IMG_H / image.shape[0], IMG_W / image.shape[1])
                new_h = max(1, int(image.shape[0] * scale))
                new_w = max(1, int(image.shape[1] * scale))
                ys = (np.arange(new_h) / scale).astype(int).clip(0, image.shape[0] - 1)
                xs = (np.arange(new_w) / scale).astype(int).clip(0, image.shape[1] - 1)
                image = image[ys][:, xs]  # nearest, keeps colours inside the palette
            self.maps.append(fit_canvas(quantize(image, self.palette)))

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        return self.maps[idx]


def to_input(indices, palette_t):
    """Class map to the RGBA tensor the encoder reads."""
    opaque = (indices != TRANSPARENT)
    rgb = palette_t[indices.clamp(max=PALETTE_SIZE - 1)].permute(0, 3, 1, 2)
    alpha = opaque.unsqueeze(1).float()
    return torch.cat([rgb * alpha, alpha], 1)


class VectorQuantizer(nn.Module):
    """EMA codebook with dead-entry restarts.

    With the plain gradient version only the winning entry is ever updated, so entries
    that start outside the encoder's output region never win and never move: the last
    run ended at perplexity 14.7 out of 512 nominal codes. EMA tracks the encoder
    independently of the optimizer learning rate, and restarting starved entries onto
    live encoder outputs puts the dead ones back into the competition.
    """

    def __init__(self):
        super().__init__()
        embed = torch.randn(N_CODES, EMBED_DIM) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.ones(N_CODES))
        self.register_buffer("ema_embed", embed.clone())

    def forward(self, z):
        flat = z.permute(0, 2, 3, 1).reshape(-1, EMBED_DIM)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.embed.t()
                + self.embed.pow(2).sum(1))
        idx = dist.argmin(1)
        q = self.embed[idx].view(z.shape[0], CODE_H, CODE_W, EMBED_DIM).permute(0, 3, 1, 2)

        if self.training:
            self._ema_update(flat, idx)

        loss = COMMIT_COST * F.mse_loss(z, q.detach())
        q = z + (q - z).detach()  # straight-through
        return q, loss, idx.view(z.shape[0], CODE_H, CODE_W)

    @torch.no_grad()
    def _ema_update(self, flat, idx):
        onehot = F.one_hot(idx, N_CODES).type(flat.dtype)
        self.cluster_size.mul_(EMA_DECAY).add_(onehot.sum(0), alpha=1 - EMA_DECAY)
        self.ema_embed.mul_(EMA_DECAY).add_(onehot.t() @ flat, alpha=1 - EMA_DECAY)

        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + EMA_EPS) / (n + N_CODES * EMA_EPS) * n
        self.embed.copy_(self.ema_embed / smoothed.unsqueeze(1))

        dead = self.cluster_size < DEAD_CODE_THRESHOLD
        if dead.any():
            picks = torch.randint(flat.shape[0], (int(dead.sum()),), device=flat.device)
            self.embed[dead] = flat[picks]
            self.ema_embed[dead] = flat[picks]
            self.cluster_size[dead] = 1.0

    def lookup(self, idx):
        return self.embed[idx].permute(0, 3, 1, 2)

    def perplexity(self, codes):
        p = torch.bincount(codes.flatten(), minlength=N_CODES).float()
        p = p[p > 0] / p.sum()
        return torch.exp(-(p * p.log()).sum()).item()


class PixelVQVAE(L.LightningModule):
    def __init__(self, palette):
        super().__init__()
        self.register_buffer("palette_t", palette)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, EMBED_DIM, 3, padding=1),
        )
        self.quantizer = VectorQuantizer()
        # PixelShuffle upsamples by predicting each subpixel separately, unlike a
        # transposed convolution that smears one kernel across the whole block.
        self.decoder = nn.Sequential(
            nn.Conv2d(EMBED_DIM, 256, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256 * 4, 3, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128 * 4, 3, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, PALETTE_SIZE + 1, 1),
        )

    def forward(self, indices):
        q, vq_loss, codes = self.quantizer(self.encoder(to_input(indices, self.palette_t)))
        return self.decoder(q), vq_loss, codes

    def training_step(self, batch, batch_idx):
        logits, vq_loss, _ = self(batch)
        ce = F.cross_entropy(logits, batch)
        self.log_dict({"ce": ce, "vq": vq_loss}, prog_bar=True)
        return ce + vq_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=2e-4)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kh, kw = self.weight.shape
        self.mask[:, :, kh // 2, kw // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, kh // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class GatedResBlock(nn.Module):
    """tanh * sigmoid gating with a residual skip. Plain stacked ReLU convolutions gave
    a shallow receptive field over the 12x36 grid and no path for gradients to skip."""

    def __init__(self, hidden, dilation=1):
        super().__init__()
        self.conv = MaskedConv2d("B", hidden, 2 * hidden, 3,
                                 padding=dilation, dilation=dilation)
        self.out = nn.Conv2d(hidden, hidden, 1)
        self.drop = nn.Dropout(PRIOR_DROPOUT)

    def forward(self, x):
        a, b = self.conv(x).chunk(2, dim=1)
        return x + self.drop(self.out(torch.tanh(a) * torch.sigmoid(b)))


class CodePrior(L.LightningModule):
    def __init__(self, hidden=PRIOR_HIDDEN):
        super().__init__()
        self.embed = nn.Embedding(N_CODES, hidden)
        self.first = MaskedConv2d("A", hidden, hidden, 5, padding=2)
        self.blocks = nn.Sequential(*[
            GatedResBlock(hidden, PRIOR_DILATIONS[i % len(PRIOR_DILATIONS)])
            for i in range(PRIOR_BLOCKS)])
        self.head = nn.Sequential(nn.ReLU(), nn.Conv2d(hidden, hidden, 1),
                                  nn.ReLU(), nn.Conv2d(hidden, N_CODES, 1))

    def forward(self, codes):
        x = self.first(self.embed(codes).permute(0, 3, 1, 2))
        return self.head(self.blocks(x))

    def training_step(self, batch, batch_idx):
        loss = F.cross_entropy(self(batch), batch)
        self.log("prior_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-4)

    @torch.no_grad()
    def sample(self, n, device, temperature=TEMPERATURE, top_k=TOP_K):
        """Sampling is 432 sequential steps and a masked convolution conditions only on
        what came before, so one improbable draw corrupts every cell after it in raster
        order. Truncating to the top k and cooling the distribution removes the tail
        that produced the scattered debris."""
        codes = torch.zeros(n, CODE_H, CODE_W, dtype=torch.long, device=device)
        for y in range(CODE_H):
            for x in range(CODE_W):
                logits = self(codes)[:, :, y, x] / temperature
                if top_k:
                    cutoff = logits.topk(min(top_k, logits.shape[1]), dim=1).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                codes[:, y, x] = torch.multinomial(logits.softmax(1), 1).squeeze(1)
        return codes


def save_indices(indices, palette, path):
    idx = indices.cpu().numpy()
    out = np.zeros((*idx.shape, 4), np.uint8)
    opaque = idx != TRANSPARENT
    out[opaque, :3] = palette[idx[opaque]]
    out[opaque, 3] = 255
    iio.imwrite(path, out)


if __name__ == "__main__":
    dataset = CloudDataset(DATA_DIR)
    palette_np = dataset.palette.astype(np.uint8)
    palette_t = torch.from_numpy(dataset.palette.astype(np.float32) / 255.0)
    print("loaded %d sprites, palette %d colours" % (len(dataset), len(palette_np)), flush=True)

    if VQVAE_CHECKPOINT and os.path.exists(VQVAE_CHECKPOINT):
        model = PixelVQVAE.load_from_checkpoint(VQVAE_CHECKPOINT, palette=palette_t)
        print("reused VQ-VAE from %s" % VQVAE_CHECKPOINT, flush=True)
    else:
        model = PixelVQVAE(palette_t)
        L.Trainer(max_epochs=EPOCHS, log_every_n_steps=10).fit(
            model, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True))
    model.eval()
    device = model.device

    maps = torch.stack(dataset.maps)
    with torch.no_grad():
        all_codes = torch.cat([model(maps[i:i + 32].to(device))[2].cpu()
                               for i in range(0, len(maps), 32)])
    print("codebook entries in use: %d / %d, perplexity %.1f"
          % (all_codes.unique().numel(), N_CODES, model.quantizer.perplexity(all_codes)), flush=True)

    prior = CodePrior()
    L.Trainer(max_epochs=PRIOR_EPOCHS, log_every_n_steps=10).fit(
        prior, DataLoader(all_codes, batch_size=BATCH_SIZE, shuffle=True))
    prior.eval().to(device)

    os.makedirs(OUT_DIR, exist_ok=True)
    with torch.no_grad():
        shown = maps[::max(1, len(dataset) // 12)][:12]
        logits, _, _ = model(shown.to(device))
        for i, recon in enumerate(logits.argmax(1).cpu()):
            save_indices(shown[i], palette_np, os.path.join(OUT_DIR, "orig_%02d.png" % i))
            save_indices(recon, palette_np, os.path.join(OUT_DIR, "recon_%02d.png" % i))

        codes = prior.sample(N_SAMPLES, device)
        samples = model.decoder(model.quantizer.lookup(codes)).argmax(1).cpu()
        for i, sample in enumerate(samples):
            save_indices(sample, palette_np, os.path.join(OUT_DIR, "sample_%02d.png" % i))

    print("wrote 12 pairs and %d samples to %s" % (N_SAMPLES, OUT_DIR))
