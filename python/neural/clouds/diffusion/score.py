"""Score a sprite with the diffusion model itself, no discriminator.

The model only ever learned to answer one question: given a canvas with pixels erased,
what belonged in them. So that is how you ask it about a finished sprite. Erase a random
tenth, read off the probability it assigns to what actually stood there, repeat until
every pixel has been asked a few times. A pixel the model reproduces is unremarkable; one
it votes against is not.

The per-pixel result is a map of where a sprite is odd, which is what a discriminator
cannot give: that produces one number for the image and needs gradients to localize.
Averaged, it separates generated sprites from hand-drawn ones at 15 of 16 on a sample of
eight each, which is a real signal but not a confident classifier for a single sprite.

Average over opaque pixels only. Over the whole canvas the score measures how much empty
background the sprite leaves, not how plausible it is: on the training set that version
tracks sprite size almost perfectly, 0.07 for the smallest against 0.51 for the largest.
"""
import argparse
import math
import os

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

import diffusion
from data import MASK, PALETTE_SIZE, quantize

ROUNDS = 60
FRACTION = 0.1
# Full red at this surprise. Opaque pixels of a hand-drawn sprite average about 1.2, so
# the scale spans the interesting range without saturating on ordinary ones.
HEAT_CEILING = 2.5


@torch.no_grad()
def surprise_map(model, x, rounds=ROUNDS, fraction=FRACTION):
    """Mean -log p(the class that is there | everything else), per pixel.

    Each round erases a different random subset, so a pixel is scored several times over
    the run and against a different neighbourhood each time. Erasing one pixel at a time
    would be exact and cost 6912 forward passes per sprite."""
    total = torch.zeros(x.shape, device=x.device, dtype=torch.float)
    asked = torch.zeros_like(total)
    step = round((1 - 2 / math.pi * math.acos(fraction)) * diffusion.TIMESTEPS)
    t = torch.full((x.size(0),), max(1, step), dtype=torch.long, device=x.device)
    for _ in range(rounds):
        selected = torch.rand(x.shape, device=x.device) < fraction
        log_probs = F.log_softmax(model(x.masked_fill(selected, MASK), t)[:, :MASK], 1)
        actual = log_probs.gather(1, x.unsqueeze(1)).squeeze(1)
        total += torch.where(selected, -actual, torch.zeros_like(actual))
        asked += selected.float()
    return total / asked.clamp(min=1)


def score(indices, maps):
    return np.array([m[a < PALETTE_SIZE].mean() for a, m in zip(indices, maps)])


def heat(values, ceiling=HEAT_CEILING, background=90):
    """Blue where the model agrees, red where it does not."""
    level = np.clip(values / ceiling, 0, 1)
    return np.stack([40 + 215 * level,
                     45 + 30 * level,
                     70 + 50 * (1 - level)], -1).astype(np.uint8)


def load_indices(directory, palette):
    names = sorted(n for n in os.listdir(directory) if n.lower().endswith(".png"))
    planes = [quantize(iio.imread(os.path.join(directory, n)), palette) for n in names]
    return names, np.stack(planes)


def main(args):
    from train import CloudDiffusion

    model = CloudDiffusion.load_from_checkpoint(args.checkpoint, data_dir=args.data_dir)
    model.eval().to(args.device)
    palette = model.palette.cpu().numpy()

    for directory in [args.input] + ([args.reference] if args.reference else []):
        names, planes = load_indices(directory, palette)
        maps = surprise_map(model.ema, torch.from_numpy(planes).to(args.device),
                            rounds=args.rounds).cpu().numpy()
        values = score(planes, maps)
        print("%s  mean %.3f" % (directory, values.mean()), flush=True)
        for name, value in zip(names, values):
            print("   %-40s %.3f" % (name, value), flush=True)

        if args.maps:
            os.makedirs(args.maps, exist_ok=True)
            for name, plane in zip(names, maps):
                iio.imwrite(os.path.join(args.maps, name), heat(plane))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="directory of sprites to score")
    parser.add_argument("--reference", help="second directory to score for comparison")
    parser.add_argument("--maps", help="write a per-pixel heat map per sprite here")
    parser.add_argument("--checkpoint", default="ckpt_diff/final.ckpt")
    parser.add_argument("--data-dir", default="../images")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
