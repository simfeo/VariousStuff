import os
from torch import optim, nn, utils, Tensor
# from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from torch.utils.data import random_split,  DataLoader, Dataset

import torchvision.transforms.functional as TF
import numpy as np
import random

import pandas as pd
from torchvision.io import read_image
import imageio as iio
import torch
import torch.nn.functional as F
import torchvision.transforms as T

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, img_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(os.path.join(self.root_dir, self.img_dir)))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_dir, os.listdir(os.path.join(self.root_dir, self.img_dir))[idx])
        # image = read_image(img_path)
        image = iio.v3.imread(img_path)
        len_y = len(image)
        len_x = len(image[0])
        print (img_path, len(image),len(image[0]))
        if self.transform:
            # transform = T.Resize(size = (sc*len_y,sc*len_x))
            # image = transform(image)
            # n_size = 256//len_y
            image = self.transform(image)
            # image = F.interpolate(image, size=n_size)
        if self.target_transform:
            image = self.target_transform(image)
        return image

#USAGE of dataclass
#    model = GAN()
#    dataset = CustomImageDataset(os.getcwd(), "inp", transform=ToTensor())
#    train_loader = utils.data.DataLoader(dataset)
#    trainer = L.Trainer(limit_train_batches=3, max_epochs=100)
#    trainer.fit(model=model, train_dataloaders=train_loader)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 1, 28, 28]

    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)
        x = self.ct1(x)
        x = F.relu(x)
        x = self.ct2(x)
        x = F.relu(x)
        return self.conv(x)


class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=0.0002):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()
        self.validation_z = torch.randn(6, self.hparams.latent_dim)
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch
        z = torch.randn(real_imgs.size(0), self.hparams.latent_dim)
        z = z.type_as(real_imgs)

 
        if (optimizer_idx ==0):
           fake_imgs = self.generator(z) # I have tried self(z) as well
           y_hat = self.discriminator(fake_imgs)
            
            
           y = torch.ones(real_imgs.size(0),1)
           y = y.type_as(real_imgs)
           g_loss = self.adversarial_loss(y_hat, y)
            
           log_dict = {"g_loss":g_loss}
           return {"loss": g_loss, "progress_bar": log_dict, "log": log_dict}
        
        # train the discriminator max log(D(x)) + log(1-D(G(z)))
        if (optimizer_idx ==1):
            # how well can it label as real and how well can it label as fake
            y_hat_real = self.discriminator(real_imgs)
            y_real = torch.ones(real_imgs.size(0),1)
            y_real = y_real.type_as(real_imgs)
            
            real_loss = self.adversarial_loss(y_hat_real, y_real)
            
            y_hat_fake = self.discriminator(self(z).detach())
            
            y_fake = torch.zeros(real_imgs.size(0),1)
            y_fake = y_fake.type_as(real_imgs)
            
            fake_loss = self.adversarial_loss(y_hat_fake, y_fake)
            
            d_loss = (real_loss+ fake_loss) / 2
            log_dict = {"d_loss":d_loss}
            return {"loss": d_loss, "progress_bar": log_dict, "log": log_dict}
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return [opt_g, opt_d],[] # empty list for schedulers
    
    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.lin1.weight)
        sample_imgs = self(z).cpu()
        print('epoch', self.current_epoch)
        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0,:,:], cmap='gray_r', interpolation='none')
            plt.title('Generated Data')
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        plt.show()
        
    def on_epoch_end(self):
        self.plot_imgs();

if __name__ == "__main__":
    dm = MNISTDataModule()
    model = GAN()
    trainer = pl.Trainer(max_epochs=20, accelerator="auto")
    trainer.fit(model,dm)
