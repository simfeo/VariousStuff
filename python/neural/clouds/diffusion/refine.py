"""Second pass over a finished sample: remask its defects and let the model repaint them.

Each reverse step predicts every masked pixel independently of the others settling with
it, so when hundreds land at once a few sit off the sprite or leave a gap inside it, and
nothing later in the process revises them. Handing the finished canvas back with only
those pixels masked puts the model in the situation it was trained on, a correct image
with holes punched in it, and it decides them against the whole neighbourhood.

Targeted rather than remasking a random fraction. A random pass clears the same defects,
but it repaints healthy pixels too, and at settings strong enough to clear all of them it
flattens the sprite past the sources: 0.74 of opaque pixels equal to their right
neighbour against 0.51 in the dataset. Targeted leaves flatness, colour count and area
unchanged to the second decimal.
"""
import argparse
import math
import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

import diffusion
from data import MASK, PALETTE_SIZE, quantize, to_rgba

MAX_SPECK = 3
# Holes run smaller than specks and a large one is usually real: measured over 8 samples,
# 26 of 33 enclosed gaps were a single pixel and the only one above 4 was a 15 pixel
# parting between two lobes of the cloud.
MAX_HOLE = 4
DILATE = 1
ROUNDS = 3


def speck_mask(indices, max_size=MAX_SPECK):
    """Opaque blobs too small to be part of the cloud."""
    out = np.zeros(indices.shape, bool)
    for i, plane in enumerate(indices):
        labels, _ = ndimage.label(plane < PALETTE_SIZE)
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        out[i] = np.isin(labels, np.flatnonzero((counts > 0) & (counts <= max_size)))
    return out


def hole_mask(indices, max_size=MAX_HOLE):
    """Transparent gaps enclosed by the sprite.

    A transparent region touching the frame is the canvas the sprite sits on, however
    small the sprite leaves it, so reaching the border is what separates background from
    a hole rather than any size threshold."""
    out = np.zeros(indices.shape, bool)
    for i, plane in enumerate(indices):
        labels, count = ndimage.label(plane >= PALETTE_SIZE)
        border = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
        sizes = np.bincount(labels.ravel())
        enclosed = [j for j in range(1, count + 1)
                    if j not in border and sizes[j] <= max_size]
        out[i] = np.isin(labels, enclosed)
    return out


def defect_mask(indices, max_speck=MAX_SPECK, max_hole=MAX_HOLE, dilate=DILATE):
    defects = speck_mask(indices, max_speck) | hole_mask(indices, max_hole)
    if dilate:
        # The ring around a defect is suspect too: a speck is often the tip of an edge
        # that frayed, and repainting it alone leaves the fray.
        defects = np.stack([ndimage.binary_dilation(d, iterations=dilate) for d in defects])
    return defects


@torch.no_grad()
def refine(model, x, rounds=ROUNDS, max_speck=MAX_SPECK, max_hole=MAX_HOLE,
           dilate=DILATE, temperature=1.0):
    """Repeat until nothing is left to fix. Repainting can expose a defect the previous
    round hid, and each pass is one forward call rather than a whole reverse process."""
    for _ in range(rounds):
        selected = torch.from_numpy(defect_mask(x.cpu().numpy(), max_speck, max_hole,
                                                dilate)).to(x.device)
        if not selected.any():
            break
        fraction = max(selected.float().mean().item(), 1e-4)
        # The model is conditioned on how much of the canvas it is meant to be missing,
        # so the timestep has to be the one whose schedule masks this fraction.
        step = round((1 - 2 / math.pi * math.acos(min(fraction, 1.0))) * diffusion.TIMESTEPS)
        t = torch.full((x.size(0),), max(1, step), dtype=torch.long, device=x.device)

        logits = model(x.masked_fill(selected, MASK), t)[:, :MASK]
        probs = F.softmax(logits / temperature, 1).permute(0, 2, 3, 1).reshape(-1, MASK)
        painted = torch.multinomial(probs, 1).reshape(x.shape)
        x = torch.where(selected, painted, x)
    return x


def main(args):
    from train import CloudDiffusion

    model = CloudDiffusion.load_from_checkpoint(args.checkpoint, data_dir=args.data_dir)
    model.eval().to(args.device)
    palette = model.palette.cpu().numpy()

    names = sorted(n for n in os.listdir(args.input) if n.lower().endswith(".png"))
    maps = np.stack([quantize(iio.imread(os.path.join(args.input, n)), palette)
                     for n in names])
    fixed = refine(model.ema, torch.from_numpy(maps).to(args.device),
                   rounds=args.rounds, max_speck=args.max_speck, max_hole=args.max_hole,
                   dilate=args.dilate, temperature=args.temperature)

    os.makedirs(args.out, exist_ok=True)
    for name, plane in zip(names, fixed.cpu().numpy()):
        iio.imwrite(os.path.join(args.out, name), to_rgba(plane, palette))
    print("refined %d images into %s" % (len(names), args.out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="directory of generated sprites")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="ckpt_diff/final.ckpt")
    parser.add_argument("--data-dir", default="../images")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--max-speck", type=int, default=MAX_SPECK)
    parser.add_argument("--max-hole", type=int, default=MAX_HOLE)
    parser.add_argument("--dilate", type=int, default=DILATE)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
