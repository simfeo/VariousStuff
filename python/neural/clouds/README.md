# Cloud sprite generation

Four attempts at the same task: generate pixel-art cloud sprites in the style of an
existing set of 523 hand-drawn PNGs with a hard alpha edge.

| folder | approach | outcome |
|---|---|---|
| `autoencoders/` | linear, convolutional and variational autoencoders | reconstructs, but sampling the latent gives blurry blobs with soft alpha |
| `gan/` | DCGAN on 256x256 and on 4-channel input | unstable, mode collapse |
| `vqvae/` | VQ-VAE with a PixelCNN prior over the code grid | works, see `vqvae/README.md` |

## Data

`images/` holds the training set: 523 RGBA sprites cropped to their content, so sizes
vary from 8x33 to 48x144 and alpha is strictly binary. The autoencoder and VQ-VAE
scripts read it through `DATA_DIR` and run as they are.

The GANs are the exception. They were trained on the same sprites padded to a square
128x128 or 256x256, which is not kept here, so their `data_dir` has to be repointed at a
set prepared that way.

Sprites larger than 48x144 are downscaled with nearest neighbour, so no colour outside
the palette can appear.

## Running

Everything here is plain PyTorch plus Lightning, no configuration files: constants live
at the top of each script and are edited in place. Run a script from its own directory,
since `DATA_DIR` is relative. Checkpoints go to `lightning_logs/version_N/`, which is
also how scripts refer back to each other, so the version numbers in the code only mean
something next to the runs that produced them. Trained weights are not stored in this
repository.
