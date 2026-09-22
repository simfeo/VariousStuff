"""UNet that reads a partly masked index map and predicts every pixel's class.

Input is discrete, so there is no tanh anywhere and no normalization of pixel values:
classes go through an embedding table, exactly as tokens would.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import N_CLASSES

EMBED_DIM = 64
BASE_CH = 128
TIME_DIM = 256


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.cos(), args.sin()], -1)


class ResBlock(nn.Module):
    """GroupNorm rather than BatchNorm: samples in a batch sit at different noise levels,
    so batch statistics mix distributions that have nothing to do with each other."""

    def __init__(self, in_ch, out_ch, time_dim=TIME_DIM):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """At the bottleneck only. A cloud's lit top and shadowed underside have to agree
    across the whole sprite, which convolutions of this depth cannot see."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(n, 3, c, h * w).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), -1)
        return x + self.out((v @ attn.transpose(1, 2)).reshape(n, c, h, w))


class UNet(nn.Module):
    @staticmethod
    def upsample(in_ch, out_ch):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                             nn.Conv2d(in_ch, out_ch, 3, padding=1))

    def __init__(self, base=BASE_CH):
        super().__init__()
        self.embed = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_DIM, TIME_DIM), nn.SiLU(), nn.Linear(TIME_DIM, TIME_DIM))
        self.stem = nn.Conv2d(EMBED_DIM, base, 3, padding=1)

        self.down1 = ResBlock(base, base)
        self.pool1 = nn.Conv2d(base, base, 4, stride=2, padding=1)       # 24x72
        self.down2 = ResBlock(base, base * 2)
        self.pool2 = nn.Conv2d(base * 2, base * 2, 4, stride=2, padding=1)  # 12x36

        self.mid1 = ResBlock(base * 2, base * 2)
        self.attn = SelfAttention(base * 2)
        self.mid2 = ResBlock(base * 2, base * 2)

        # Nearest-neighbour then convolve. Stacked ConvTranspose2d is what collapsed the
        # GAN in ../gan onto a fixed lattice, and there is no reason to invite it back.
        self.up2 = self.upsample(base * 2, base * 2)
        self.res2 = ResBlock(base * 4, base)
        self.up1 = self.upsample(base, base)
        self.res1 = ResBlock(base * 2, base)

        self.head = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, N_CLASSES, 1))

    def forward(self, x, t):
        t_emb = self.time_mlp(timestep_embedding(t, TIME_DIM))
        h0 = self.stem(self.embed(x).permute(0, 3, 1, 2))

        s1 = self.down1(h0, t_emb)
        s2 = self.down2(self.pool1(s1), t_emb)
        m = self.mid2(self.attn(self.mid1(self.pool2(s2), t_emb)), t_emb)

        d2 = self.res2(torch.cat([self.up2(m), s2], 1), t_emb)
        d1 = self.res1(torch.cat([self.up1(d2), s1], 1), t_emb)
        return self.head(d1)  # logits over classes, one distribution per pixel
