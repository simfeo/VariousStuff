"""DCGAN over 256x256 RGBA cloud sprites.

Every output pixel of the previous version sat on an odd x and an even y: five stacked
ConvTranspose2d layers with no normalization had collapsed the generator onto a periodic
lattice instead of a shape. The upsampling, the normalization and the loss below are all
aimed at that failure. See README.md for the full list.
"""
import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_SIZE = 256
CHANNELS = 4
LATENT_DIM = 128
BATCH_SIZE = 16
LR = 2e-4
# The discriminator is trained against 0.9 rather than 1.0 for real images. It wins this
# matchup easily, and once it is confident the generator's gradient through the logistic
# loss all but disappears.
REAL_LABEL = 0.9
SAMPLE_EVERY = 10
DATA_DIR = "images_256"
OUT_DIR = "out_gan"


def weights_init(module):
    """DCGAN initialization. Kaiming suits one network descending one loss, not two that
    have to stay balanced against each other."""
    name = module.__class__.__name__
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear) and "Generator" not in name:
        nn.init.normal_(module.weight, 0.0, 0.02)


class UpBlock(nn.Module):
    """PixelShuffle instead of ConvTranspose2d.

    Five stacked transposed convolutions collapsed the output onto a fixed lattice.
    Here each subpixel of a 2x2 block comes from its own channel, so no position in the
    block is privileged and nothing accumulates across layers."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 512 * 8 * 8)
        self.bn0 = nn.BatchNorm2d(512)
        self.ups = nn.Sequential(
            UpBlock(512, 256),   # 16
            UpBlock(256, 128),   # 32
            UpBlock(128, 64),    # 64
            UpBlock(64, 32),     # 128
            UpBlock(32, 16),     # 256
        )
        self.out = nn.Conv2d(16, CHANNELS, 3, padding=1)

    def forward(self, z):
        x = self.lin1(z).view(-1, 512, 8, 8)
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
            # First block unnormalized: it reads the raw input distribution, which is
            # part of what separates a real sprite from a generated one.
            *block(CHANNELS, 64, norm=False),   # 128
            *block(64, 128),                    # 64
            *block(128, 256),                   # 32
            *block(256, 256),                   # 16
            *block(256, 512),                   # 8
            *block(512, 512),                   # 4
            # A convolution down to a single value instead of Linear(256*8*8, 1), which
            # held two thirds of the parameters and invited memorizing 523 sprites.
            nn.Conv2d(512, 1, 4),
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)  # logits, the loss applies the sigmoid


class RGBAImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(f for f in os.listdir(root_dir)
                                  if f.lower().endswith((".png", ".jpg")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.image_files[idx])).convert("RGBA")
        return self.transform(image) if self.transform else image


class GAN(L.LightningModule):
    def __init__(self, latent_dim=LATENT_DIM, lr=LR, data_dir=DATA_DIR):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.criterion = nn.BCEWithLogitsLoss()
        # Buffer, so the preview noise follows the model onto the GPU and stays the same
        # across epochs: the sample grid then shows how one fixed z evolves.
        self.register_buffer("validation_z", torch.randn(8, latent_dim))
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real = batch
        n = real.size(0)
        opt_g, opt_d = self.optimizers()

        real_labels = torch.full((n, 1), REAL_LABEL, device=self.device)
        fake_labels = torch.zeros(n, 1, device=self.device)

        # Discriminator first, on its own noise: reusing the generator step's z trains it
        # against exactly the samples the generator has just been fitted to.
        z = torch.randn(n, self.hparams.latent_dim, device=self.device)
        fake = self(z).detach()
        d_loss = 0.5 * (self.criterion(self.discriminator(real), real_labels)
                        + self.criterion(self.discriminator(fake), fake_labels))
        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()

        z = torch.randn(n, self.hparams.latent_dim, device=self.device)
        gen = self(z)
        g_loss = self.criterion(self.discriminator(gen), torch.ones(n, 1, device=self.device))
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
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS),
        ])
        dataset = RGBAImageDataset(root_dir=self.hparams.data_dir, transform=transform)
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, persistent_workers=True, drop_last=True)

    def on_train_epoch_end(self):
        """The previous version logged samples from on_validation_epoch_end with no
        validation dataloader defined, so the hook never fired and 1000 epochs ran blind."""
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
        grid = torchvision.utils.make_grid(sample, nrow=4, normalize=True, value_range=(-1, 1))
        if self.logger is not None:
            self.logger.experiment.add_image("generated", grid, self.current_epoch)
        os.makedirs(OUT_DIR, exist_ok=True)
        torchvision.utils.save_image(grid, os.path.join(OUT_DIR, "epoch_%04d.png" % epoch))


def save_rgba(tensor, path, alpha_threshold=0.5):
    """tanh output back to an RGBA PNG.

    Not save_image(normalize=True): without value_range it rescales by the tensor's own
    min and max across all four channels, putting colour and alpha on an arbitrary scale."""
    img = ((tensor.cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()
    rgba = (img * 255).astype(np.uint8)
    # The sources have two alpha values; nothing in the loss forces a hard edge, so it is
    # imposed here. The cut sits mid-range because faint alpha covers the empty canvas.
    opaque = img[..., 3] > alpha_threshold
    rgba[..., 3] = np.where(opaque, 255, 0)
    rgba[~opaque] = 0
    Image.fromarray(rgba, mode="RGBA").save(path)


def train(args):
    model = GAN(data_dir=args.data_dir)
    trainer = L.Trainer(max_epochs=args.epochs, log_every_n_steps=10,
                        enable_progress_bar=False)
    trainer.fit(model)
    trainer.save_checkpoint(args.checkpoint)
    print("saved %s" % args.checkpoint, flush=True)


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
    parser.add_argument("--generate", action="store_true", help="sample a trained checkpoint")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--checkpoint", default="gan_256.ckpt")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.generate:
        generate(args)
    else:
        train(args)
