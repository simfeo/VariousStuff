import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
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
import math

# dimm_x = 93
# dimm_y = 261


dimm_x = 125
dimm_y = 269


def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

class MineSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        if len(args):
            super(MineSequential, self).__init__(args)
        else:
            super(MineSequential, self).__init__()

    def append(self, layer):
        if hasattr(layer,'reset_parameters'):
            layer.reset_parameters()
            layer = MSRInitializer(layer)
        super().append(layer)

    def reset(self):
        for module in self: # Iterate through modules
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MineSequential()
        self.model.append( nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1) )  # Output: 63x134
        self.model.append( nn.LeakyReLU(0.2) )
        self.model.append( nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1) )  # Output: 32x67
        self.model.append( nn.LeakyReLU(0.2) )
        self.model.append( nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) )  # Output: 16x34
        self.model.append( nn.LeakyReLU(0.2) )
        self.model.append( nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) ) # Output: 8x17
        self.model.append( nn.LeakyReLU(0.2) )
        self.model.append( nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) ) # Output: 4x9
        self.model.append( nn.LeakyReLU(0.2) )
        self.model.append( nn.Flatten() )
        self.model.append( nn.Linear(6144, 1) )  # Corrected input size
        self.model.append( nn.Sigmoid() )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.model = MineSequential()
        self.model.append( nn.Linear(latent_dim, 16 * 4 ) )
        self.model.append( nn.ReLU() )
        self.model.append( nn.Linear(16*4, 32 * 4 ) )
        self.model.append( nn.ReLU() )
        self.model.append( nn.Unflatten(1, (32, 1, 4)) )  # Reshape
        self.model.append( nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3, padding=0) )
        self.model.append( nn.ReLU() )
        self.model.append( nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=0) )
        self.model.append( nn.ReLU() )
        self.model.append( nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=0) )
        self.model.append( nn.ReLU() )
        self.model.append( nn.ConvTranspose2d(16, 8, kernel_size=5, stride=2, padding=0) )
        self.model.append( nn.ReLU() )
        self.model.append( nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2, padding=0) )
        self.model.append( nn.ReLU() )
        self.model.append( nn.Conv2d(4, 4, kernel_size=1, padding=0) )
        self.model.append( nn.Tanh() )
        

    def forward(self, x):
        x = self.model(x)
        return x


  
# Assuming Generator and Discriminator classes are defined elsewhere
# For example:
# class Generator(nn.Module): ...
# class Discriminator(nn.Module): ...

class RGBAImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.real_len = len(self.image_files)

    def __len__(self):
        return self.real_len * 50

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[ idx % self.real_len ])
        # image = Image.open(img_path).convert('RGBA')  # Ensure RGBA format
        image = Image.open(img_path).convert('RGBA')  # Ensure RGBA format
        if self.transform:
            image = self.transform(image)
        return image

class GAN(L.LightningModule):
    def __init__(self, latent_dim=1, lr=0.0002, data_dir='images'):
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
        # print("real_images shape:", real_images.shape)

        # Access optimizers
        opt_g, opt_d = self.optimizers()

        # Sample noise
        z = torch.randn(real_images.shape[0], self.hparams.latent_dim, device=self.device)

        # Train Generator
        fake_images = self(z)
        # print("fake_images shape:", fake_images.shape)

        y_hat = self.discriminator(fake_images)
        # print("y_hat shape:", y_hat.shape)

        y = torch.ones(real_images.size(0), 1, device=self.device)
        # print("y shape:", y.shape)
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
        if d_loss == 50.0:
            self.discriminator.reset()
        elif d_loss > 0.0001:
            opt_d.zero_grad()
            self.manual_backward(d_loss)
            opt_d.step()
        with open(Path(self.logger.log_dir)/"f_out.txt","a") as control_log:
            control_log.write("{}|{:.4f}|{:.4f}\n".format(self.current_epoch, g_loss, d_loss))


    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        # transform = transforms.Compose([
        #     transforms.Resize((128, 128)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))  # Normalize for 4 channels
        # ])
        transform = transforms.Compose([
            transforms.CenterCrop((dimm_x,dimm_y)),
            transforms.RandomAffine(0,translate=(5/dimm_y,5/dimm_x)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
        ])
        dataset = RGBAImageDataset(root_dir=self.hparams.data_dir, transform=transform)
        return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    def on_validation_epoch_end(self):
        z = self.validation_z.to(self.device)
        sample = self(z)
        grid = torchvision.utils.make_grid(sample)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        # step = 2
        # if self.current_epoch > 1 and (self.current_epoch % step) == 0:
        #     torch.save(self, Path(self.logger.log_dir)/"checkpoints"/f"epoch={self.current_epoch}.ckpt")
        pass

    def on_save_checkpoint(self, checkpoint):
        step = 20
        current_epoch = self.current_epoch + 1
        if current_epoch > 1 and (current_epoch % step) == 0:
            torch.save(checkpoint, Path(self.logger.log_dir)/"checkpoints"/f"epoch={self.current_epoch}.ckpt")
        return super().on_save_checkpoint(checkpoint)



if __name__ == '__main__':
    # Your main code here
    if len (sys.argv) == 1:
        # checkpoint = "lightning_logs/version_30/checkpoints/epoch=199.ckpt"
        # gan_model = GAN.load_from_checkpoint(checkpoint) 
        gan_model = GAN()

        # checkpoint_callback = ModelCheckpoint(
        #     dirpath= 'checkpoints/',  # Directory to save checkpoints
        #     filename='model-{epoch:02d}-{val_loss:.2f}',  # Filename format
        #     save_top_k=-1,  # Save all checkpoints
        #     every_n_epochs=2  # Save every 100 epochs
        # )
        # trainer = L.Trainer(max_epochs=1000, callbacks=[checkpoint_callback])
        trainer = L.Trainer(max_epochs=2)
        trainer.fit(gan_model)
    elif sys.argv[1] == "--generate":
        for i in range(19,1000,20):
            print ("genarating")
            # 1. Load the Trained Generator (as you've done)
            checkpoint = "lightning_logs/version_2/checkpoints/epoch=1-step=3272.ckpt"
            # checkpoint = f"lightning_logs/version_15/checkpoints/epoch={i}.ckpt"
            autoencoder = GAN.load_from_checkpoint(checkpoint)  # Assuming GAN is the class name
            generator = autoencoder.generator
            generator.eval()  # Set to evaluation mode

            now = datetime.datetime.now()
            time_stamp = now.strftime('%Y-%m-%d_%H-%M-%S')
            
            counter = len(sys.argv) != 2 and int(sys.argv[2]) or 1
            while counter > 0:
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
                torchvision.utils.save_image(
                            generated_image,
                            Path(os.getcwd()) / f"tgt/{i:0>3}_fake-{time_stamp}-{counter}.png",
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


# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Input: dimm_yxdimm_x with 4 channels
#         self.conv1 = nn.Conv2d(4, 16, kernel_size=4, stride=2, padding=1)  # Output: 130x46
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1) # Output: 65x23
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output: 32x11
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # Output: 16x5
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # Output: 8x2
#         self.lin1 = nn.Linear(256 * 8 * 2, 1)  # Adjusted linear layer input size

#     def forward(self, x):
#         x = F.leaky_relu(self.conv1(x), 0.2)
#         x = F.leaky_relu(self.conv2(x), 0.2)
#         x = F.leaky_relu(self.conv3(x), 0.2)
#         x = F.leaky_relu(self.conv4(x), 0.2)
#         x = F.leaky_relu(self.conv5(x), 0.2)
#         x = x.view(-1, 256 * 8 * 2)  # Flatten before linear layer
#         return torch.sigmoid(self.lin1(x))  # Sigmoid for probability output



# class Generator(nn.Module):
#     def __init__(self, latent_dim):
#         super().__init__()
#         # Calculate the initial size needed to get to dimm_yxdimm_x
#         # We'll start with a smaller feature map and upsample
#         self.lin1 = nn.Linear(latent_dim, 32 * 4 * 3)  # Adjust for initial size
#         self.ct1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1) # Upsample to 8x6
#         self.ct2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=3, padding=1)  # Upsample to 24x18
#         self.ct3 = nn.ConvTranspose2d(16, 8, kernel_size=5, stride=4, padding=1)   # Upsample to dimm_xx69
#         self.ct4 = nn.ConvTranspose2d(8, 4, kernel_size=5, stride=3, padding=1)   # Upsample to 277x205
#         self.conv = nn.Conv2d(4, 4, kernel_size=1, padding=0)  # Output 3 channels (RGB), adjust padding

#     def forward(self, x):
#         x = self.lin1(x)
#         x = F.relu(x)
#         x = x.view(-1, 32, 4, 3) # Reshape to the calculated initial size

#         x = self.ct1(x)
#         x = F.relu(x)

#         x = self.ct2(x)
#         x = F.relu(x)

#         x = self.ct3(x)
#         x = F.relu(x)

#         x = self.ct4(x)
#         x = F.relu(x)

#         x = self.conv(x)
#         return torch.tanh(x[:, :, :dimm_y, :dimm_x])  # Crop to dimm_yxdimm_x
