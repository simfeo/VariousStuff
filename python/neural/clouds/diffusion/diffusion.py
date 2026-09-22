"""Discrete diffusion with an absorbing state, over palette indices.

Gaussian diffusion adds continuous noise to RGB values, which is the regression setting
that ../gan lost in: 0.03% of its pixels landed on a palette colour. Here the forward
process instead replaces pixels with MASK, and the model classifies what belonged there.
Every intermediate state is a legal index map, so the output is pixel art by construction
rather than by thresholding afterwards.

The schedule is cosine and the reverse step unmasks a uniformly random subset. Ranking by
confidence instead, as MaskGIT does, empties the canvas here: 77% of the training pixels
are transparent, so from a fully masked start the model predicts transparent nearly
everywhere and is most confident exactly there. Those pixels settle first, which makes
the next prediction more transparent still. Gumbel noise on the ranking only slows the
collapse down, measured at 0.06 opaque against the sources' 0.23; a random subset is
unbiased and reaches 0.18.
"""
import math

import torch
import torch.nn.functional as F

from data import MASK

TIMESTEPS = 256


def mask_ratio(r):
    """Fraction masked at normalized time r. Zero at r=0, everything at r=1."""
    return torch.cos(math.pi / 2 * (1 - r))


def q_sample(x0, t, timesteps=TIMESTEPS):
    """Mask each pixel independently with the probability this timestep calls for."""
    ratio = mask_ratio(t.float() / timesteps)[:, None, None]
    masked = torch.rand_like(x0, dtype=torch.float) < ratio
    return torch.where(masked, torch.full_like(x0, MASK), x0), masked


def loss(model, x0, timesteps=TIMESTEPS):
    t = torch.randint(1, timesteps + 1, (x0.size(0),), device=x0.device)
    xt, masked = q_sample(x0, t, timesteps)
    logits = model(xt, t)
    # Only masked positions carry signal: everywhere else the answer is already in the
    # input, and training on those collapses the loss to copying.
    per_pixel = F.cross_entropy(logits, x0, reduction="none")
    return (per_pixel * masked).sum() / masked.sum().clamp(min=1)


@torch.no_grad()
def sample(model, shape, device, steps=32, timesteps=TIMESTEPS, temperature=1.0):
    """Reverse process from a fully masked canvas.

    Fewer reverse steps than forward timesteps is deliberate: each step re-predicts every
    masked pixel from scratch, so the count trades sampling time against quality without
    changing what the model was trained on."""
    x = torch.full(shape, MASK, dtype=torch.long, device=device)
    for i in range(steps):
        t_now = 1 - i / steps
        t_next = 1 - (i + 1) / steps
        t = torch.full((shape[0],), max(1, round(t_now * timesteps)),
                       dtype=torch.long, device=device)

        logits = model(x, t)[:, :MASK]  # MASK is never a prediction, only an input state
        probs = F.softmax(logits / temperature, 1)
        flat = probs.permute(0, 2, 3, 1).reshape(-1, probs.size(1))
        pred = torch.multinomial(flat, 1).reshape(shape)

        masked = x == MASK
        x = torch.where(masked, pred, x)
        if t_next <= 0:
            break

        keep_masked = int(mask_ratio(torch.tensor(t_next)).item() * masked[0].numel())
        if keep_masked == 0:
            continue
        score = torch.rand(shape, device=device).masked_fill(~masked, float("inf"))
        cut = score.flatten(1).topk(keep_masked, largest=False).indices
        remask = torch.zeros_like(score.flatten(1), dtype=torch.bool)
        remask.scatter_(1, cut, True)
        x = x.masked_fill(remask.reshape(shape), MASK)
    return x
