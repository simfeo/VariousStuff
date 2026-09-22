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
LATENT_DIM = 64
EPOCHS = 400
BATCH_SIZE = 16
BETA = 1.0
KL_WARMUP_EPOCHS = 100
OUT_DIR = "out_vae"
N_SAMPLES = 16


def load_sprite(path):
    """Fit an RGBA sprite into the 48x144 canvas. 89 percent of the set already fits,
    so most images are padded rather than resized. Oversized ones use nearest: alpha
    here is strictly 0 or 255, and any smooth filter turns the hard edges into fringes
    that the model would then learn to reproduce."""
    image = iio.imread(path)
    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    h, w = tensor.shape[1:]
    scale = min(IMG_H / h, IMG_W / w, 1.0)
    if scale < 1.0:
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        tensor = F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)

    # Transparent pixels carry arbitrary RGB in these sprites, so premultiply to keep
    # the loss from fitting colours that are never visible.
    tensor[:3] *= tensor[3:]

    canvas = torch.zeros(4, IMG_H, IMG_W)
    top, left = (IMG_H - tensor.shape[1]) // 2, (IMG_W - tensor.shape[2]) // 2
    canvas[:, top:top + tensor.shape[1], left:left + tensor.shape[2]] = tensor
    return canvas


class CloudDataset(Dataset):
    """Every sprite is centred. Position is deliberately not a latent factor here: the
    previous run randomised placement, so sampling mixed incompatible positions and
    produced clouds broken into scattered pieces."""

    def __init__(self, img_dir):
        names = sorted(n for n in os.listdir(img_dir) if n.lower().endswith(".png"))
        self.sprites = [load_sprite(os.path.join(img_dir, n)) for n in names]

    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, idx):
        return self.sprites[idx]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),   # 24x72
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),  # 12x36
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2), # 6x18
        )
        self.fc_mu = nn.Linear(128 * 6 * 18, LATENT_DIM)
        self.fc_logvar = nn.Linear(128 * 6 * 18, LATENT_DIM)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 128 * 6 * 18)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 4, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.conv(self.fc(z).view(-1, 128, 6, 18))


class CloudVAE(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def training_step(self, batch, batch_idx):
        mu, logvar = self.encoder(batch)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        x_hat = self.decoder(z)

        # Both terms are summed over the sample and averaged over the batch, so the
        # reconstruction and KL magnitudes stay comparable and beta means what it says.
        recon = F.mse_loss(x_hat, batch, reduction="none").flatten(1).sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()

        # Ramping the KL in avoids posterior collapse, where the decoder learns to
        # ignore z entirely in the first epochs and never recovers.
        beta = BETA * min(1.0, (self.current_epoch + 1) / KL_WARMUP_EPOCHS)
        loss = recon + beta * kl

        self.log_dict({"recon": recon, "kl": kl, "loss": loss}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def save_png(tensor, path):
    img = tensor.detach().cpu().clamp(0, 1)
    alpha = img[3:].clamp(min=1e-6)
    img = torch.cat([(img[:3] / alpha).clamp(0, 1), img[3:]])  # undo premultiply
    iio.imwrite(path, (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


if __name__ == "__main__":
    dataset = CloudDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("loaded %d sprites" % len(dataset), flush=True)

    model = CloudVAE()
    L.Trainer(max_epochs=EPOCHS, log_every_n_steps=10).fit(model, loader)

    os.makedirs(OUT_DIR, exist_ok=True)
    model.eval()
    device = model.device
    with torch.no_grad():
        originals = torch.stack(dataset.sprites)
        shown = originals[::max(1, len(dataset) // 12)][:12]
        mu, _ = model.encoder(shown.to(device))
        for i, recon in enumerate(model.decoder(mu).cpu()):
            save_png(shown[i], os.path.join(OUT_DIR, "orig_%02d.png" % i))
            save_png(recon, os.path.join(OUT_DIR, "recon_%02d.png" % i))

        # A VAE's prior is N(0, I) by construction, so sampling needs no latent stats.
        z = torch.randn(N_SAMPLES, LATENT_DIM, device=device)
        for i, sample in enumerate(model.decoder(z).cpu()):
            save_png(sample, os.path.join(OUT_DIR, "sample_%02d.png" % i))

        mus = torch.cat([model.encoder(originals[i:i + 64].to(device))[0].cpu()
                         for i in range(0, len(originals), 64)])
        active = (mus.std(0) > 0.1).sum().item()
        print("active latent dims: %d / %d" % (active, LATENT_DIM))

    print("wrote 12 pairs and %d samples to %s" % (N_SAMPLES, OUT_DIR))
