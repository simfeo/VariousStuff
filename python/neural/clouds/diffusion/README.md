# Discrete diffusion over palette indices

## Why not Gaussian diffusion

The usual denoising diffusion model adds Gaussian noise to RGB values and learns to
subtract it. That puts it in exactly the setting the GANs in `../gan` lost in: continuous
regression over colour. Measured against the 552 colours of the training set, the
regression GAN landed 0.03% of its pixels on a palette colour and never once repeated a
colour in two neighbouring pixels, where the sources do so 45% of the time. It drew a
dithered gradient shaped like a cloud.

Here the forward process replaces pixels with an absorbing `MASK` state instead of
perturbing them, and the model classifies what belonged in each masked position. Every
intermediate state is a legal index map, so palette membership and flat fills are
properties of the representation, not of training longer. This is the same reason
`../vqvae` beat every GAN variant, and the same reason `../gan/GAN_palette.py` caught up
once its generator classified rather than regressed.

## The pieces

| file | what it holds |
|---|---|
| `data.py` | palette construction, quantization, the 48x144 dataset with pixel shifts |
| `model.py` | UNet over class embeddings, timestep conditioning, attention at the bottleneck |
| `diffusion.py` | cosine mask schedule, the training objective, confidence-based sampling |
| `train.py` | Lightning module, EMA, contact sheets, CLI |

## Running

```
python train.py --epochs 600
python train.py --generate --count 8 --checkpoint ckpt_diff/epoch=0299.ckpt --out samples
```

Contact sheets land in `out_diff/epoch_NNNN.png` every 25 epochs, composited over grey:
a transparent sprite on a white page hides everything light, which is most of a cloud.
Checkpoints are kept every 50 epochs and the last one is not the one to reach for. Over
the 48x144 GAN's 600 epochs the best samples came from epoch 550, and over the palette
GAN's from epoch 500.

## Sampling

`--steps` is the number of reverse steps and defaults to 32 against 256 forward
timesteps. Each reverse step re-predicts every still-masked pixel, settles a random
subset of them and returns the rest to `MASK`.

Keeping the most confident pixels instead, which is MaskGIT's recipe, empties the canvas.
77% of the training pixels are transparent, so from a fully masked start the model
predicts transparent nearly everywhere and is most confident exactly there; those settle
first and the next prediction is more transparent still. Measured at epoch 50, confidence
ranking produced 0.0 opaque pixels, 0.06 with Gumbel noise on the ranking, against 0.18
for a random subset and 0.23 in the sources.

`--temperature` below 1.0 sharpens the per-pixel distribution: fewer stray colours,
less variety between samples.

## What one 600 epoch run produced

Measured over 8 samples per checkpoint. `agree` is the share of opaque pixels equal to
their right neighbour, `comp` the mean number of connected components per sprite.

| | palette | agree | colours | comp | opaque |
|---|---|---|---|---|---|
| epoch 50 | 100% | 0.16 | 249 | 58.5 | 0.12 |
| epoch 200 | 100% | 0.45 | 214 | 10.1 | 0.29 |
| epoch 350 | 100% | 0.41 | 245 | 3.2 | 0.19 |
| epoch 500 | 100% | 0.46 | 171 | 2.6 | 0.24 |
| epoch 600 | 100% | 0.38 | 236 | 3.6 | 0.22 |
| sources | 100% | 0.51 | 552 | 1.0 | 0.52 |

Palette membership is 100% from the first sampled epoch, which is the point of the
discrete formulation and not something training earned. What training buys is coherence:
58 components at epoch 50 means the sprite was a constellation of specks in roughly the
right place, and it falls monotonically for 450 epochs as the model learns to condition
on the pixels already unmasked.

Epoch 500 is the best checkpoint and the last one is not, which held for all three
architectures tried on this dataset. `../gan/GAN_palette.py` at its own epoch 500 scores
0.47 agree and 1.1 components on the same measurement, so the GAN still wins on silhouette
coherence while the two tie on flatness and size. Diffusion trained far more calmly: no
collapse, no restarts, near-monotonic improvement throughout.
