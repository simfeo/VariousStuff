# VQ-VAE with a PixelCNN prior

The approach that produced usable sprites. Two stages:

1. A VQ-VAE compresses a 48x144 sprite to a 12x36 grid of codes drawn from a 512 entry
   codebook. Colour is handled as classification over a 256 colour palette built from
   the training set, plus one extra class for "transparent" - so the silhouette comes
   out discrete by construction and no alpha thresholding is needed.
2. A gated PixelCNN learns the distribution of those code grids. Sampling it and
   decoding gives a new sprite.

## Files

| file | role |
|---|---|
| `train_clouds_pixel.py` | the current model: both stages, training and sampling |
| `train_clouds_vqvae.py` | the earlier version, kept because its prior is undilated and shows what changed |
| `continue_prior.py` | resumes the prior from a checkpoint for more epochs |
| `sample_prior.py` | samples a saved prior without training anything |
| `clean_alpha.py` | post-process: binarize alpha and drop detached specks from decoder output |

`continue_prior.py` and `sample_prior.py` both `import train_clouds_pixel`, so they must
sit next to it.

## What the settings cost

The constants at the top of `train_clouds_pixel.py` are the result of the runs, not
defaults. The ones that mattered:

- **EMA codebook with dead-entry restarts.** With a plain gradient codebook only the
  winning entry is ever updated, entries that start outside the encoder's output region
  never win and never move, and the run ended using 14.7 of 512 nominal codes.
- **Dilated prior blocks, cycling 1, 2, 4, 8.** Undilated blocks add one cell of context
  each, so 12 of them reached only about 14 of the 36 columns, and the model kept
  starting a second cloud it could not see. Cycling the dilation covers the full width at
  the same depth and parameter count.
- **400 prior epochs, not 600.** The loss is flat from 400 to 600 (0.0529 to 0.0527) and
  only drops again by 800, which tracked copying of the training sprites rising from 6.4
  to 14.7 percent agreement.
- **Temperature 0.8 and top-k 35 when sampling.** Sampling is 432 sequential steps and a
  masked convolution conditions only on what came before, so one improbable draw corrupts
  every cell after it in raster order. Truncating the tail removed the scattered debris.

## Before running

`DATA_DIR` reads `../images` and is relative, so run the scripts from this directory.

`VQVAE_CHECKPOINT` still names a `lightning_logs` version from the original runs, and no
weights are stored here: set it to `None` to train the VQ-VAE from scratch instead of
reusing one. The same goes for `CHECKPOINT` in `continue_prior.py` and `sample_prior.py`,
which have nothing to read until a prior has been trained.
