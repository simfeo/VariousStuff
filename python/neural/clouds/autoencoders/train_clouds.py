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

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(32 * 32, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 32 * 32))


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


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
            n_size = 256//len_y
            image = self.transform(image)
            image = F.interpolate(image, size=n_size)
        if self.target_transform:
            image = self.target_transform(image)
        return image

autoencoder = LitAutoEncoder(encoder, decoder)

# tgt_tr = lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)

dataset = CustomImageDataset(os.getcwd(), "inp", transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)

trainer = L.Trainer(limit_train_batches=3, max_epochs=100)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)