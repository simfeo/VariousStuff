# DCGAN over 256x256 RGBA sprites

## The failure this was rewritten around

The previous generator did not produce a blurry cloud, it produced a lattice. Measured
over the last 20 outputs of the old run, every single opaque pixel sat on an odd x and an
even y, 2% of the canvas, 8 distinct colours. That is a pure checkerboard artifact: the
model had learned a periodic pattern with roughly the right fraction of opaque pixels and
nothing else.

## What changed

| | before | after |
|---|---|---|
| upsampling | 5x `ConvTranspose2d` k=4 s=2 | `Conv2d` + `PixelShuffle` |
| normalization | none anywhere | `BatchNorm2d` in both, except the discriminator's first block |
| loss | `sigmoid` then `binary_cross_entropy` | `BCEWithLogitsLoss` on raw logits |
| weight init | PyTorch default | `normal(0, 0.02)`, `normal(1, 0.02)` for BatchNorm |
| discriminator head | `Linear(256*8*8, 1)` | `Conv2d(512, 1, 4)` |
| noise for the D step | reused the generator's `z` | drawn fresh |
| step order | generator, then discriminator | discriminator, then generator |
| real label | 1.0 | 0.9 |
| sample preview | `on_validation_epoch_end`, never fired | `on_train_epoch_end`, every 10 epochs |
| saving | `save_image(normalize=True)` | explicit `value_range=(-1, 1)`, alpha thresholded |
| data | sprites pasted at (0, 0) | `prepare_256.py`, centred |

Transposed convolutions are what generated the lattice, so they are gone: PixelShuffle
predicts the subpixels of each 2x2 output block from separate channels, and no position
inside the block is privileged.

The normalization, the initialization and the logit loss all address the same thing from
different sides. Without them the discriminator wins within a few epochs, and a confident
discriminator passes almost no gradient back through a saturated sigmoid.

## Running

```
python prepare_256.py ../images images_256     # once
python GAN_256_4_chanels.py --epochs 400
python GAN_256_4_chanels.py --generate --count 16 --checkpoint gan_256.ckpt
```

Sample grids land in `out_gan/epoch_NNNN.png` during training, which is the point of
having them: this architecture can still collapse, and the grids show when.

## The 48x144 version

A centred sprite covers about 2% of a 256x256 canvas, so most of what both networks model
there is emptiness. `GAN_48x144.py` trains at the sprites' own aspect ratio instead, the
frame `../vqvae` uses, where a cloud fills roughly a third. It reuses the pieces above and
pads in the dataset, so it reads `../images` with no prepared copy on disk.

It is the better of the two: cumulus silhouettes with separate humps rather than one
smooth mound. Over 600 epochs the best samples came from epoch 550, not the last one, and
neither the sample diversity nor the speckle moved monotonically, so it writes a
checkpoint every 50 epochs and the last one is not the one to reach for.

## What neither version solves

Both are regressions over continuous RGBA, and the sprites are pixel art. Measured over
16 generated 48x144 sprites against the 552 colours of the training set: 0.03% of the
generated pixels land exactly on a palette colour, and no opaque pixel ever equals its
right-hand neighbour, against 45% in the sources. The output is a dithered gradient in the
shape of a cloud. `../vqvae` scores 100% and 32% on the same two measures because its
decoder classifies over the palette instead of regressing colour, which is a property of
the architecture and not something more training reaches.
