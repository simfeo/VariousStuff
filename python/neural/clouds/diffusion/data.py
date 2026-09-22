"""Sprites as palette index maps on a 48x144 canvas.

Self-contained rather than imported from ../gan: the two trees share the idea of
quantizing to a palette and nothing else, and a sys.path insert to reach across
folders would be worse than eighty lines.
"""
import collections
import os

import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_H, IMG_W = 48, 144
PALETTE_SIZE = 256
TRANSPARENT = PALETTE_SIZE
# The absorbing state. Not a colour: it means "not decided yet" during sampling, and the
# forward process replaces pixels with it.
MASK = PALETTE_SIZE + 1
N_CLASSES = PALETTE_SIZE + 2
SHIFT = 2
DATA_DIR = "../images"


def build_palette(img_dir, names):
    counts = collections.Counter()
    for name in names:
        image = iio.imread(os.path.join(img_dir, name))
        counts.update(map(tuple, image[image[..., 3] > 128][:, :3]))
    return np.array([c for c, _ in counts.most_common(PALETTE_SIZE)], dtype=np.int32)


def quantize(image, palette):
    rgb = image[..., :3].astype(np.int32)  # int32: squared differences overflow int16
    idx = ((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(-1).argmin(-1)
    idx = idx.astype(np.int64)
    idx[image[..., 3] <= 128] = TRANSPARENT
    return idx


def to_rgba(indices, palette):
    """Index map back to an RGBA image. MASK renders transparent, which only shows up
    when an intermediate sampling step is saved."""
    opaque = indices < PALETTE_SIZE
    rgba = np.zeros(indices.shape + (4,), np.uint8)
    rgba[..., :3][opaque] = palette[indices[opaque]]
    rgba[..., 3] = np.where(opaque, 255, 0)
    return rgba


class SpriteIndexDataset(Dataset):
    def __init__(self, root_dir=DATA_DIR, shift=SHIFT):
        self.shift = shift
        names = sorted(n for n in os.listdir(root_dir) if n.lower().endswith(".png"))
        self.palette = build_palette(root_dir, names)
        self.sprites = []
        for name in names:
            image = iio.imread(os.path.join(root_dir, name))
            h, w = image.shape[:2]
            if h > IMG_H or w > IMG_W:
                scale = min(IMG_H / h, IMG_W / w)
                ys = (np.arange(max(1, int(h * scale))) / scale).astype(int).clip(0, h - 1)
                xs = (np.arange(max(1, int(w * scale))) / scale).astype(int).clip(0, w - 1)
                image = image[ys][:, xs]
            self.sprites.append(quantize(image, self.palette))

    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, idx):
        sprite = self.sprites[idx]
        h, w = sprite.shape
        top, left = (IMG_H - h) // 2, (IMG_W - w) // 2
        # 51 of 523 sprites fill the frame to within two rows, so the offset is clamped
        # per sprite rather than globally.
        dy = np.random.randint(-min(self.shift, top), min(self.shift, IMG_H - h - top) + 1)
        dx = np.random.randint(-min(self.shift, left), min(self.shift, IMG_W - w - left) + 1)

        canvas = np.full((IMG_H, IMG_W), TRANSPARENT, dtype=np.int64)
        canvas[top + dy:top + dy + h, left + dx:left + dx + w] = sprite
        return torch.from_numpy(canvas)
