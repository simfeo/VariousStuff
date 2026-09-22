"""Pad the cloud sprites into centred 256x256 RGBA canvases.

The earlier preparation pasted every sprite at (0, 0), so across the whole set no
opaque pixel ever appeared below y=83: the discriminator could separate real from fake
on canvas position alone. Centring removes that shortcut.

Usage: prepare_256.py [src_dir] [dst_dir]
"""
import os
import sys

import imageio.v3 as iio
import numpy as np

CANVAS = 256
SRC_DEFAULT = "../images"
DST_DEFAULT = "images_256"


def centre(image):
    h, w = image.shape[:2]
    if h > CANVAS or w > CANVAS:
        scale = min(CANVAS / h, CANVAS / w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        ys = (np.arange(new_h) / scale).astype(int).clip(0, h - 1)
        xs = (np.arange(new_w) / scale).astype(int).clip(0, w - 1)
        image = image[ys][:, xs]  # nearest, so no colour outside the source appears
        h, w = image.shape[:2]

    canvas = np.zeros((CANVAS, CANVAS, 4), np.uint8)
    top, left = (CANVAS - h) // 2, (CANVAS - w) // 2
    canvas[top:top + h, left:left + w] = image
    # RGB under fully transparent pixels is never displayed but is still fed to the
    # network, so zero it instead of leaving whatever the source happened to store.
    canvas[canvas[..., 3] < 128] = 0
    canvas[..., 3] = np.where(canvas[..., 3] >= 128, 255, 0)
    return canvas


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else SRC_DEFAULT
    dst = sys.argv[2] if len(sys.argv) > 2 else DST_DEFAULT
    os.makedirs(dst, exist_ok=True)

    names = sorted(n for n in os.listdir(src) if n.lower().endswith(".png"))
    fill = []
    for name in names:
        out = centre(iio.imread(os.path.join(src, name)))
        iio.imwrite(os.path.join(dst, name), out)
        fill.append((out[..., 3] > 0).mean())

    fill = np.array(fill)
    print("wrote %d canvases to %s" % (len(names), dst))
    print("opaque fraction: min %.3f median %.3f max %.3f"
          % (fill.min(), np.median(fill), fill.max()))
