"""Resume the code prior from a checkpoint for extra epochs and sample from it.

Used to compare an early-stopped prior against the same prior trained longer, without
retraining anything else: the VQ-VAE and the palette stay byte for byte identical.
"""
import os
import sys

import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader

import train_clouds_pixel as tp

CHECKPOINT = "keep/prior_dilated_400ep.ckpt"
TOTAL_EPOCHS = 500
OUT_DIR = "out_pixel_500"
N_SAMPLES = 16

if __name__ == "__main__":
    dataset = tp.CloudDataset(tp.DATA_DIR)
    palette_np = dataset.palette.astype(np.uint8)
    palette_t = torch.from_numpy(dataset.palette.astype(np.float32) / 255.0)

    model = tp.PixelVQVAE.load_from_checkpoint(tp.VQVAE_CHECKPOINT, palette=palette_t).eval()
    device = model.device
    maps = torch.stack(dataset.maps)
    with torch.no_grad():
        codes = torch.cat([model(maps[i:i + 32].to(device))[2].cpu()
                           for i in range(0, len(maps), 32)])

    prior = tp.CodePrior()
    # ckpt_path restores the optimizer and the epoch counter, so max_epochs is the new
    # total and the run picks up exactly where the saved one stopped.
    L.Trainer(max_epochs=TOTAL_EPOCHS, log_every_n_steps=10).fit(
        prior, DataLoader(codes, batch_size=tp.BATCH_SIZE, shuffle=True),
        ckpt_path=CHECKPOINT)
    prior.eval().to(device)

    os.makedirs(OUT_DIR, exist_ok=True)
    with torch.no_grad():
        sampled = prior.sample(N_SAMPLES, device)
        images = model.decoder(model.quantizer.lookup(sampled)).argmax(1).cpu()
    for i, img in enumerate(images):
        tp.save_indices(img, palette_np, os.path.join(OUT_DIR, "sample_%02d.png" % i))
    print("wrote %d samples to %s" % (N_SAMPLES, OUT_DIR))
