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
CODE_H, CODE_W = 6, 18
EMBED_DIM = 64
N_CODES = 256
COMMIT_COST = 0.25
EPOCHS = 600
PRIOR_EPOCHS = 400
BATCH_SIZE = 16
OUT_DIR = "out_vqvae"
N_SAMPLES = 16


def load_sprite(path):
    image = iio.imread(path)
    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    h, w = tensor.shape[1:]
    scale = min(IMG_H / h, IMG_W / w, 1.0)
    if scale < 1.0:
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        tensor = F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
    tensor[:3] *= tensor[3:]

    canvas = torch.zeros(4, IMG_H, IMG_W)
    top, left = (IMG_H - tensor.shape[1]) // 2, (IMG_W - tensor.shape[2]) // 2
    canvas[:, top:top + tensor.shape[1], left:left + tensor.shape[2]] = tensor
    return canvas


class CloudDataset(Dataset):
    def __init__(self, img_dir):
        names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(".png"))
        self.sprites = [load_sprite(os.path.join(img_dir, n)) for n in names]

    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, idx):
        return self.sprites[idx]


class VectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.codebook = nn.Embedding(N_CODES, EMBED_DIM)
        self.codebook.weight.data.uniform_(-1.0 / N_CODES, 1.0 / N_CODES)

    def forward(self, z):
        flat = z.permute(0, 2, 3, 1).reshape(-1, EMBED_DIM)
        dist = (flat.pow(2).sum(1, keepdim=True)
                - 2 * flat @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))
        idx = dist.argmin(1)
        q = self.codebook(idx).view(z.shape[0], CODE_H, CODE_W, EMBED_DIM).permute(0, 3, 1, 2)

        loss = F.mse_loss(q, z.detach()) + COMMIT_COST * F.mse_loss(z, q.detach())
        # Straight-through: quantization has no gradient, so the decoder's gradient is
        # copied onto the encoder output unchanged.
        q = z + (q - z).detach()
        return q, loss, idx.view(z.shape[0], CODE_H, CODE_W)

    def lookup(self, idx):
        return self.codebook(idx).permute(0, 3, 1, 2)


class VQVAE(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, EMBED_DIM, 3, padding=1),
        )
        self.quantizer = VectorQuantizer()
        self.decoder = nn.Sequential(
            nn.Conv2d(EMBED_DIM, 128, 3, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 4, 3, padding=1), nn.Sigmoid(),
        )

    def training_step(self, batch, batch_idx):
        q, vq_loss, _ = self.quantizer(self.encoder(batch))
        recon = F.mse_loss(self.decoder(q), batch)
        loss = recon + vq_loss
        self.log_dict({"recon": recon, "vq": vq_loss}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=2e-4)


class MaskedConv2d(nn.Conv2d):
    """Mask type A hides the current pixel, type B keeps it. This is what makes the
    prior autoregressive over the code grid in raster order."""

    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))
        _, _, kh, kw = self.weight.shape
        self.mask[:, :, kh // 2, kw // 2 + (mask_type == "B"):] = 0
        self.mask[:, :, kh // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class CodePrior(L.LightningModule):
    """A VQ-VAE decoder only maps codes to pixels. Drawing codes independently at random
    gives noise, so generation needs a learned distribution over whole code grids."""

    def __init__(self, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(N_CODES, hidden)
        layers = [MaskedConv2d("A", hidden, hidden, 5, padding=2), nn.ReLU()]
        for _ in range(6):
            layers += [MaskedConv2d("B", hidden, hidden, 3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(hidden, N_CODES, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, codes):
        return self.net(self.embed(codes).permute(0, 3, 1, 2))

    def training_step(self, batch, batch_idx):
        loss = F.cross_entropy(self(batch), batch)
        self.log("prior_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=3e-4)

    @torch.no_grad()
    def sample(self, n, device):
        codes = torch.zeros(n, CODE_H, CODE_W, dtype=torch.long, device=device)
        for y in range(CODE_H):
            for x in range(CODE_W):
                probs = self(codes)[:, :, y, x].softmax(1)
                codes[:, y, x] = torch.multinomial(probs, 1).squeeze(1)
        return codes


def save_png(tensor, path):
    img = tensor.detach().cpu().clamp(0, 1)
    alpha = img[3:].clamp(min=1e-6)
    img = torch.cat([(img[:3] / alpha).clamp(0, 1), img[3:]])
    iio.imwrite(path, (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


if __name__ == "__main__":
    dataset = CloudDataset(DATA_DIR)
    print("loaded %d sprites" % len(dataset), flush=True)

    vqvae = VQVAE()
    L.Trainer(max_epochs=EPOCHS, log_every_n_steps=10).fit(
        vqvae, DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True))

    vqvae.eval()
    device = vqvae.device
    originals = torch.stack(dataset.sprites)

    with torch.no_grad():
        all_codes = torch.cat([vqvae.quantizer(vqvae.encoder(originals[i:i + 64].to(device)))[2]
                               for i in range(0, len(originals), 64)])
    used = all_codes.unique().numel()
    print("codebook entries in use: %d / %d" % (used, N_CODES), flush=True)

    prior = CodePrior()
    L.Trainer(max_epochs=PRIOR_EPOCHS, log_every_n_steps=10).fit(
        prior, DataLoader(all_codes.cpu(), batch_size=BATCH_SIZE, shuffle=True))
    prior.eval().to(device)

    os.makedirs(OUT_DIR, exist_ok=True)
    with torch.no_grad():
        shown = originals[::max(1, len(dataset) // 12)][:12]
        q, _, _ = vqvae.quantizer(vqvae.encoder(shown.to(device)))
        for i, recon in enumerate(vqvae.decoder(q).cpu()):
            save_png(shown[i], os.path.join(OUT_DIR, "orig_%02d.png" % i))
            save_png(recon, os.path.join(OUT_DIR, "recon_%02d.png" % i))

        codes = prior.sample(N_SAMPLES, device)
        for i, sample in enumerate(vqvae.decoder(vqvae.quantizer.lookup(codes)).cpu()):
            save_png(sample, os.path.join(OUT_DIR, "sample_%02d.png" % i))

    print("wrote 12 pairs and %d samples to %s" % (N_SAMPLES, OUT_DIR))
