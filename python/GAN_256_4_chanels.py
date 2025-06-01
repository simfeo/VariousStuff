import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset  # Import Dataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio

from PIL import Image
import os

from torch.utils.data import random_split,  DataLoader, Dataset

import torch.nn as nn
import torch.nn.functional as F

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
    gan_model = GAN()
    trainer = L.Trainer()
    trainer.fit(gan_model)



# class CustomImageDataset(Dataset):
#     def __init__(self, root_dir, img_dir, transform=None, target_transform=None):
#         self.root_dir = root_dir
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(os.listdir(os.path.join(self.root_dir, self.img_dir)))

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root_dir, self.img_dir, os.listdir(os.path.join(self.root_dir, self.img_dir))[idx])
#         # image = read_image(img_path)
#         image = iio.v3.imread(img_path)
#         # len_y = len(image)  # Removed unnecessary variables
#         # len_x = len(image[0])
#         # print (img_path, len(image),len(image[0]))  # Removed print statement
#         if self.transform:
#             # transform = T.Resize(size = (sc*len_y,sc*len_x))
#             # image = transform(image)
#             # n_size = 256//len_y
#             image = self.transform(image)
#             # image = F.interpolate(image, size=n_size)
#         if self.target_transform:
#             image = self.target_transform(image)
#         return image


    # def train_dataloader(self):
    #     transform = transforms.Compose([
    #         transforms.Resize(64),  # Or whatever size your generator expects
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    #     ])
    #     dataset = CustomImageDataset(root_dir=self.hparams.data_dir, img_dir = self.hparams.img_dir, transform=transform)
    #     return DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    # def train_dataloader(self):
    #     transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
    #     ])
    #     dataset = datasets.ImageFolder(root=self.hparams.data_dir, transform=transform)
    #     return DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)


# class CustomImageDataset(Dataset):
#     def __init__(self, root_dir, img_dir, transform=None, target_transform=None):
#         self.root_dir = root_dir
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(os.listdir(os.path.join(self.root_dir, self.img_dir)))

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.root_dir, self.img_dir, os.listdir(os.path.join(self.root_dir, self.img_dir))[idx])
#         # image = read_image(img_path)
#         image = iio.v3.imread(img_path)
#         len_y = len(image)
#         len_x = len(image[0])
#         print (img_path, len(image),len(image[0]))
#         if self.transform:
#             # transform = T.Resize(size = (sc*len_y,sc*len_x))
#             # image = transform(image)
#             # n_size = 256//len_y
#             image = self.transform(image)
#             # image = F.interpolate(image, size=n_size)
#         if self.target_transform:
#             image = self.target_transform(image)
#         return image

# class GAN_old(L.LightningModule):
#     def __init__(self, latent_dim=100, lr=0.0002, data_dir='inp'):
#         super().__init__()
#         self.save_hyperparameters()
#         self.generator = Generator(latent_dim=self.hparams.latent_dim)
#         self.discriminator = Discriminator()
#         self.validation_z = torch.randn(6, self.hparams.latent_dim)
#         self.data_dir = data_dir

#     def forward(self, z):
#         return self.generator(z)

#     def adversarial_loss(self, y_hat, y):
#         return F.binary_cross_entropy(y_hat, y)

#     def training_step(self, batch, batch_idx, optimizer_idx):
#         real_images = batch[0]
#         real_images = real_images.type(torch.float32)

#         # Sample noise
#         z = torch.randn(real_images.shape[0], self.hparams.latent_dim, device=self.device)

#         # Train Generator
#         if optimizer_idx == 0:
#             fake_images = self(z)
#             y_hat = self.discriminator(fake_images)
#             y = torch.ones(real_images.size(0), 1, device=self.device)
#             g_loss = self.adversarial_loss(y_hat, y)
#             self.log("g_loss", g_loss, prog_bar=True)
#             return g_loss

#         # Train Discriminator
#         if optimizer_idx == 1:
#             # Real images
#             y_hat_real = self.discriminator(real_images)
#             y_real = torch.ones(real_images.size(0), 1, device=self.device)
#             real_loss = self.adversarial_loss(y_hat_real, y_real)

#             # Fake images
#             fake_images = self(z).detach()
#             y_hat_fake = self.discriminator(fake_images)
#             y_fake = torch.zeros(real_images.size(0), 1, device=self.device)
#             fake_loss = self.adversarial_loss(y_hat_fake, y_fake)

#             d_loss = (real_loss + fake_loss) / 2
#             self.log("d_loss", d_loss, prog_bar=True)
#             return d_loss

#     def configure_optimizers(self):
#         lr = self.hparams.lr
#         opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
#         opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
#         return [opt_g, opt_d], []

#     @property
#     def automatic_optimization(self) -> bool:
#         return False

#     def train_dataloader(self):
#         transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
#         ])
#         dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
#         return DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

#     def on_validation_epoch_end(self):
#         z = self.validation_z.to(self.device)
#         sample = self(z)
#         grid = make_grid(sample)
#         self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

