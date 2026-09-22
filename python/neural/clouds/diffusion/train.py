"""Training and sampling entry point for the discrete cloud diffusion."""
import argparse
import copy
import datetime
import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import diffusion
from data import DATA_DIR, IMG_H, IMG_W, SpriteIndexDataset, to_rgba
from model import UNet

BATCH_SIZE = 32
LR = 2e-4
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = 500
SAMPLE_EVERY = 25
CHECKPOINT_EVERY = 50
SAMPLE_STEPS = 32
OUT_DIR = "out_diff"


class CloudDiffusion(L.LightningModule):
    def __init__(self, lr=LR, data_dir=DATA_DIR):
        super().__init__()
        self.save_hyperparameters()
        self.net = UNet()
        self.ema = copy.deepcopy(self.net).requires_grad_(False)
        self.dataset = SpriteIndexDataset(data_dir)
        self.register_buffer("palette", torch.from_numpy(self.dataset.palette))

    def training_step(self, batch, batch_idx):
        value = diffusion.loss(self.net, batch)
        self.log("loss", value, prog_bar=True)
        return value

    def on_train_batch_end(self, *_):
        decay = EMA_DECAY if self.global_step >= EMA_WARMUP_STEPS else 0.0
        with torch.no_grad():
            for ema, live in zip(self.ema.parameters(), self.net.parameters()):
                ema.lerp_(live, 1 - decay)
            for ema, live in zip(self.ema.buffers(), self.net.buffers()):
                ema.copy_(live)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.net.parameters(), lr=self.hparams.lr,
                                 weight_decay=0.01)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, persistent_workers=True, drop_last=True)

    @torch.no_grad()
    def draw(self, count, steps=SAMPLE_STEPS, temperature=1.0):
        self.ema.eval()
        indices = diffusion.sample(self.ema, (count, IMG_H, IMG_W), self.device,
                                   steps=steps, temperature=temperature)
        return to_rgba(indices.cpu().numpy(), self.palette.cpu().numpy())

    def on_train_epoch_end(self):
        epoch = self.current_epoch + 1
        print("epoch %4d  loss %.4f"
              % (epoch, self.trainer.callback_metrics["loss"].item()), flush=True)
        if epoch % SAMPLE_EVERY and epoch != 1:
            return
        images = self.draw(8)
        os.makedirs(OUT_DIR, exist_ok=True)
        write_sheet(images, os.path.join(OUT_DIR, "epoch_%04d.png" % epoch))


def write_sheet(images, path, scale=4, cols=2, pad=6, bg=90):
    """Composited over grey rather than saved with alpha. A contact sheet of transparent
    sprites viewed on a white background hides everything light, which is most of a cloud."""
    tiles = []
    for image in images:
        alpha = image[..., 3:4] / 255.0
        rgb = image[..., :3] * alpha + bg * (1 - alpha)
        tiles.append(np.repeat(np.repeat(rgb, scale, 0), scale, 1).astype(np.uint8))
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), bg, np.uint8)
    for i, tile in enumerate(tiles):
        y, x = divmod(i, cols)
        sheet[pad + y * (h + pad):pad + y * (h + pad) + h,
              pad + x * (w + pad):pad + x * (w + pad) + w] = tile
    iio.imwrite(path, sheet)


def train(args):
    model = CloudDiffusion(data_dir=args.data_dir)
    keeper = ModelCheckpoint(dirpath=args.ckpt_dir, filename="{epoch:04d}",
                             every_n_epochs=CHECKPOINT_EVERY, save_top_k=-1,
                             save_on_train_epoch_end=True)
    trainer = L.Trainer(max_epochs=args.epochs, log_every_n_steps=10,
                        enable_progress_bar=False, callbacks=[keeper])
    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(args.ckpt_dir, "final.ckpt"))
    print("saved %s" % os.path.join(args.ckpt_dir, "final.ckpt"), flush=True)


def generate(args):
    model = CloudDiffusion.load_from_checkpoint(args.checkpoint, data_dir=args.data_dir)
    model.eval().to(args.device)
    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    images = model.draw(args.count, steps=args.steps, temperature=args.temperature)
    for i, image in enumerate(images):
        iio.imwrite(Path(args.out) / ("cloud-%s-%02d.png" % (stamp, i)), image)
    print("wrote %d images to %s" % (args.count, args.out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--steps", type=int, default=SAMPLE_STEPS)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--ckpt-dir", default="ckpt_diff")
    parser.add_argument("--checkpoint", default="ckpt_diff/final.ckpt")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    generate(args) if args.generate else train(args)
