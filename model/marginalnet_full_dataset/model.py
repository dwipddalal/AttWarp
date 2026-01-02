from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_softmax(logits: torch.Tensor, dim: int = 1, eps: float = 1e-6) -> torch.Tensor:
    """Softmax with aggressive NaN/Inf guards and re-normalization."""
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    logits = logits - logits.amax(dim=dim, keepdim=True)  # stability
    p = F.softmax(logits, dim=dim)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return p / p.sum(dim=dim, keepdim=True).clamp_min(eps)


class MarginalNet(nn.Module):
    """
    Predict px (len W) and py (len H) from:
      - visual token map V in (B, Dv, hv, wv)  -> upsample to (H,W)
      - text tokens T in (B, Lt, Dt)           -> pooled -> FiLM(V)
    Heads output normalized densities along X and Y axes.
    """

    def __init__(self, d_vis_in: int, d_txt_in: int, hidden: int = 256, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.proj_v = nn.Sequential(
            nn.Conv2d(d_vis_in, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.txt_pool = nn.Sequential(
            nn.Linear(d_txt_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        # FiLM: produce (gamma, beta) per-channel
        self.film = nn.Linear(hidden, 2 * hidden)

        # Heads: separate 1D aggregators along axes
        self.head_x = nn.Sequential(  # (B, hidden, W) -> (B, W)
            nn.Conv1d(hidden, hidden, 5, padding=2, groups=1),
            nn.SiLU(),
            nn.Conv1d(hidden, 1, 1),
        )
        self.head_y = nn.Sequential(  # (B, hidden, H) -> (B, H)
            nn.Conv1d(hidden, hidden, 5, padding=2, groups=1),
            nn.SiLU(),
            nn.Conv1d(hidden, 1, 1),
        )

    def forward(
        self,
        fmap_v: torch.Tensor,
        H: int,
        W: int,
        txt_tok: torch.Tensor,
        txt_mask: torch.Tensor,
    ):
        """
        fmap_v: (B, Dv, hv, wv) visual map (from LLaVA)
        H,W: target size (match attention map)
        txt_tok: (B, Lt, Dt), txt_mask: (B, Lt, 1)
        returns px: (B, W), py: (B, H) (positive, normalized)
        """
        fmap_v = fmap_v.float()
        txt_tok = txt_tok.float()
        txt_mask = txt_mask.float()

        # 1) visual projection + upsample
        v = self.proj_v(fmap_v)  # (B, hidden, hv, wv)
        v = F.interpolate(v, size=(H, W), mode="bilinear", align_corners=False)

        # 2) text pooling + FiLM
        denom = txt_mask.sum(dim=1).clamp_min(1.0)
        t = (txt_tok * txt_mask).sum(dim=1) / denom  # (B, Dt)
        t = self.txt_pool(t)  # (B, hidden)
        gamma_beta = self.film(t)  # (B, 2*hidden)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        v = gamma * v + beta

        # 3) heads
        vx = v.mean(dim=2)  # (B, hidden, W) integrate over Y
        vy = v.mean(dim=3)  # (B, hidden, H) integrate over X
        logit_x = self.head_x(vx).squeeze(1)
        logit_y = self.head_y(vy).squeeze(1)

        px = safe_softmax(logit_x, dim=1, eps=self.eps)
        py = safe_softmax(logit_y, dim=1, eps=self.eps)
        return px, py


def mix_with_uniform(p: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        return p
    return (1 - alpha) * p + alpha / p.size(1)


def entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q = p.clamp_min(eps)
    return -(q * q.log()).sum(dim=1).mean()


