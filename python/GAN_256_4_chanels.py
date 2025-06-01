import pytorch_lightning as L
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset  # Import Dataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio

from PIL import Image
import os
import sys
from pathlib import Path

from torch.utils.data import random_split,  DataLoader, Dataset

import torch.nn as nn
import torch.nn.functional as F
import datetime

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.lin1 = nn.Linear(256 * 8 * 8, 1)  # Corrected linear layer input size

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = x.view(-1, 256 * 8 * 8)  # Flatten before linear layer
        return torch.sigmoid(self.lin1(x))  # Sigmoid for probability output


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 8*8*512) # Adjust for 256x256 output
        self.ct1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1) # Upsample to 16x16
        self.ct2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1) # Upsample to 32x32
        self.ct3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # Upsample to 64x64
        self.ct4 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)   # Upsample to 128x128
        self.ct5 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)   # Upsample to 256x256
        self.conv = nn.Conv2d(16, 4, kernel_size=3, padding=1) # Output 4 channels

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 512, 8, 8)

        x = self.ct1(x)
        x = F.relu(x)

        x = self.ct2(x)
        x = F.relu(x)

        x = self.ct3(x)
        x = F.relu(x)

        x = self.ct4(x)
        x = F.relu(x)

        x = self.ct5(x)
        x = F.relu(x)

        return torch.tanh(self.conv(x)) # Tanh for normalization to [-1, 1]

  
# Assuming Generator and Discriminator classes are defined elsewhere
# For example:
# class Generator(nn.Module): ...
# class Discriminator(nn.Module): ...

def rgb_transform(image):
    return image[:3, :, :]  # Keep only the first 3 channels (RGB)

class RGBAImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        # image = Image.open(img_path).convert('RGBA')  # Ensure RGBA format
        image = Image.open(img_path).convert('RGBA')  # Ensure RGBA format
        if self.transform:
            image = self.transform(image)
        return image

class GAN(L.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002, data_dir='images', img_dir = 'class1'):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()
        self.validation_z = torch.randn(6, self.hparams.latent_dim)
        self.automatic_optimization = False  # Set to False for manual optimization

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        # Ensure y is the same size as y_hat
        # if y.size(0) != y_hat.size(0):
        #     y = y[:y_hat.size(0)]  # Adjust y to match the size of y_hat
        return F.binary_cross_entropy(y_hat, y)


    def training_step(self, batch, batch_idx):
        real_images = batch
        real_images = real_images.type(torch.float32)
        print("real_images shape:", real_images.shape)

        # Sample noise
        z = torch.randn(real_images.shape[0], self.hparams.latent_dim, device=self.device)

        # Access optimizers
        opt_g, opt_d = self.optimizers()

        # Train Generator
        fake_images = self(z)
        print("fake_images shape:", fake_images.shape)

        y_hat = self.discriminator(fake_images)
        print("y_hat shape:", y_hat.shape)

        y = torch.ones(real_images.size(0), 1, device=self.device)
        g_loss = self.adversarial_loss(y_hat, y)

        self.log("g_loss", g_loss, prog_bar=True)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        # Train Discriminator
        y_hat_real = self.discriminator(real_images)
        y_real = torch.ones(real_images.size(0), 1, device=self.device)
        real_loss = self.adversarial_loss(y_hat_real, y_real)

        fake_images = self(z).detach()
        y_hat_fake = self.discriminator(fake_images)
        y_fake = torch.zeros(real_images.size(0), 1, device=self.device)
        fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))  # Normalize for 4 channels
        ])
        dataset = RGBAImageDataset(root_dir=self.hparams.data_dir, transform=transform)
        return DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
        


    def on_validation_epoch_end(self):
        z = self.validation_z.to(self.device)
        sample = self(z)
        grid = torchvision.utils.make_grid(sample)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


if __name__ == '__main__':
    # Your main code here
    if len (sys.argv) == 1:
        gan_model = GAN()
        trainer = L.Trainer()
        trainer.fit(gan_model)
    elif sys.argv[1] == "--generate":
        print ("genarating")
        # 1. Load the Trained Generator (as you've done)
        checkpoint = "lightning_logs/version_12/checkpoints/epoch=999-step=10000.ckpt"
        autoencoder = GAN.load_from_checkpoint(checkpoint)  # Assuming GAN is the class name
        generator = autoencoder.generator
        generator.eval()  # Set to evaluation mode

        counter = len(sys.argv) != 2 and int(sys.argv[2]) or 0
        while counter >= 0:
            # 2. Prepare the Input Noise
            latent_dim = autoencoder.hparams.latent_dim  # Get the latent dimension from the loaded model's hparams
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            generator.to(device)
            noise = torch.randn(1, latent_dim, device=device)  # Create a batch of 1 image
            # z = torch.randn(real_image.shape[0],self.hparams.latent_dim)
            # noise = torch.randn([1, 4, 256, 256], latent_dim, device=device)  # Create a batch of 1 image

            # 3. Generate the Image
            with torch.no_grad():
                generated_image = generator(noise)

            print ("generated_image shape",generated_image.shape,"N", counter)

            now = datetime.datetime.now()
            time_stamp = now.strftime('%Y-%m-%d_%H-%M-%S')
            torchvision.utils.save_image(
                        generated_image,
                        Path(os.getcwd()) / f"tgt/fake-{time_stamp}-{counter}.png",
                        padding=2,
                        normalize=True,
                    )
            counter -= 1

    else:
        print ("USAGE:")
        print ("For training:")
        print(os.path.basename(__file__))
        print("")
        print ("For generating")
        print(os.path.basename(__file__) + " --generate")
