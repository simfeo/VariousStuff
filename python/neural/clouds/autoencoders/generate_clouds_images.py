import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import lightning as L
from torchvision.transforms import ToTensor

CHECKPOINT = "lightning_logs/version_48/checkpoints/epoch=99-step=300.ckpt"
OUT_DIR = "out"
N_SAMPLES = 16
# Height of the generated clouds. Training squashed every image to 4 * H * (256 // H)
# floats, so H fixes how the flat 1024-vector is folded back into a picture.
GEN_H = 16
VIEW_W = 64

encoder = nn.Sequential(nn.Linear(32 * 32, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 32 * 32))


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


def load_training_tensor(path):
    """Repeat the exact preprocessing of train_clouds.py, squashing width to 256 // H."""
    image = iio.imread(path)
    h = image.shape[0]
    tensor = ToTensor()(image)
    return F.interpolate(tensor, size=256 // h)


def to_png(flat, h, path):
    img = flat.reshape(4, h, 256 // h).clamp(0, 1)
    img = F.interpolate(img.unsqueeze(0), size=(h, VIEW_W), mode="nearest").squeeze(0)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    iio.imwrite(path, img)


model = LitAutoEncoder.load_from_checkpoint(
    CHECKPOINT, encoder=encoder, decoder=decoder, map_location="cpu"
)
model.eval()

os.makedirs(OUT_DIR, exist_ok=True)

latents = []
with torch.no_grad():
    for name in sorted(os.listdir("inp")):
        src = load_training_tensor(os.path.join("inp", name))
        h = src.shape[1]
        flat = src.reshape(1, -1)
        z = model.encoder(flat)
        latents.append(z)
        to_png(model.decoder(z)[0], h, os.path.join(OUT_DIR, "recon_" + name))
        print("encoded %-40s h=%-4d z=%s" % (name, h, z[0].tolist()), flush=True)

latents = torch.cat(latents)
z_min, z_max = latents.min(0).values, latents.max(0).values
print("latent min", z_min.tolist())
print("latent max", z_max.tolist())

with torch.no_grad():
    for i in range(N_SAMPLES):
        z = z_min + (z_max - z_min) * torch.rand(1, 3)
        to_png(model.decoder(z)[0], GEN_H, os.path.join(OUT_DIR, "sample_%02d.png" % i))

print("wrote %d reconstructions and %d samples to %s" % (len(latents), N_SAMPLES, OUT_DIR))
