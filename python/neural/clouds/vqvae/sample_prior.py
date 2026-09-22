"""Sample from a saved code prior without training anything.

continue_prior.py resumes training before it samples, so it cannot be used to just look
at a snapshot: the 400 epoch checkpoint would be advanced past the point it was kept for.
"""
import os
import sys

import imageio.v3 as iio
import numpy as np
import torch

import train_clouds_pixel as tp

CHECKPOINT = "keep/prior_dilated_400ep.ckpt"
OUT_DIR = "out_pixel_400_gen"
N_SAMPLES = 16
SHEET_COLS = 4
SHEET_SCALE = 3
SHEET_PAD = 4


def make_sheet(images, palette, path):
    rows = (len(images) + SHEET_COLS - 1) // SHEET_COLS
    cell_h, cell_w = tp.IMG_H * SHEET_SCALE, tp.IMG_W * SHEET_SCALE
    sheet = np.zeros((rows * (cell_h + SHEET_PAD) + SHEET_PAD,
                      SHEET_COLS * (cell_w + SHEET_PAD) + SHEET_PAD, 4), np.uint8)
    # Mid grey behind the sprites, otherwise transparent and white pixels look the same.
    sheet[..., :3] = 90
    sheet[..., 3] = 255
    for n, idx in enumerate(images):
        idx = idx.cpu().numpy()
        rgba = np.zeros((*idx.shape, 4), np.uint8)
        opaque = idx != tp.TRANSPARENT
        rgba[opaque, :3] = palette[idx[opaque]]
        rgba[opaque, 3] = 255
        rgba = rgba.repeat(SHEET_SCALE, 0).repeat(SHEET_SCALE, 1)
        y = SHEET_PAD + (n // SHEET_COLS) * (cell_h + SHEET_PAD)
        x = SHEET_PAD + (n % SHEET_COLS) * (cell_w + SHEET_PAD)
        cell = sheet[y:y + cell_h, x:x + cell_w]
        alpha = rgba[..., 3:4] / 255.0
        cell[..., :3] = (rgba[..., :3] * alpha + cell[..., :3] * (1 - alpha)).astype(np.uint8)
    iio.imwrite(path, sheet)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(seed)

    dataset = tp.CloudDataset(tp.DATA_DIR)
    palette_np = dataset.palette.astype(np.uint8)
    palette_t = torch.from_numpy(dataset.palette.astype(np.float32) / 255.0)
    print("loaded %d sprites" % len(dataset), flush=True)

    model = tp.PixelVQVAE.load_from_checkpoint(tp.VQVAE_CHECKPOINT, palette=palette_t).eval()
    device = model.device

    prior = tp.CodePrior()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    prior.load_state_dict(state)
    prior.eval().to(device)
    print("loaded prior from %s on %s" % (CHECKPOINT, device), flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with torch.no_grad():
        codes = prior.sample(N_SAMPLES, device)
        images = model.decoder(model.quantizer.lookup(codes)).argmax(1).cpu()
    for i, img in enumerate(images):
        tp.save_indices(img, palette_np, os.path.join(OUT_DIR, "sample_%02d.png" % i))
    make_sheet(images, palette_np, os.path.join(OUT_DIR, "sheet.png"))
    print("wrote %d samples and sheet.png to %s" % (N_SAMPLES, OUT_DIR), flush=True)
