import os
import sys

import imageio.v3 as iio
import numpy as np
from scipy import ndimage

ALPHA_THRESHOLD = 128
# The sources themselves contain components as small as 1px, so this only removes
# what is clearly decoder noise and still costs a little genuine detail.
MIN_BLOB_PIXELS = 4


def clean(image):
    """Force alpha back to the strictly binary form the source sprites use, then drop
    detached specks. The decoder emits a few soft pixels that read as fringe and noise
    even though they are a small fraction of the canvas."""
    out = image.copy()
    mask = out[..., 3] >= ALPHA_THRESHOLD

    labels, count = ndimage.label(mask)
    if count:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        keep = np.isin(labels, np.flatnonzero(sizes >= MIN_BLOB_PIXELS) + 1)
        mask &= keep

    # Interior holes are NOT filled: 12 percent of the source sprites carry deliberate
    # 1 to 5 pixel gaps inside the cloud body, and filling them erases that style.
    out[..., 3] = np.where(mask, 255, 0)
    out[~mask] = 0
    return out


if __name__ == "__main__":
    src_dir, dst_dir = sys.argv[1], sys.argv[2]
    os.makedirs(dst_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(src_dir) if n.lower().endswith(".png"))
    for name in names:
        iio.imwrite(os.path.join(dst_dir, name), clean(iio.imread(os.path.join(src_dir, name))))
    print("cleaned %d images into %s" % (len(names), dst_dir))
