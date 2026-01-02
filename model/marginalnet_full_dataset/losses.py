from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from checkpoint_utils import cdf_from_density


def l1_cdf_loss(p_pred: torch.Tensor, p_gt: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(cdf_from_density(p_pred), cdf_from_density(p_gt))


def sym_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True)
    q = q.clamp_min(eps)
    q = q / q.sum(dim=1, keepdim=True)
    kl_pq = (p * (p.log() - q.log())).sum(dim=1).mean()
    kl_qp = (q * (q.log() - p.log())).sum(dim=1).mean()
    return 0.5 * (kl_pq + kl_qp)


def separable_recon_loss(px: torch.Tensor, py: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    px:(B,W), py:(B,H), A:(B,1,H,W) -> compare outer product to normalized A
    """
    Apos = A.clamp_min(0)
    Apos = Apos / Apos.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    P = torch.einsum("bh,bw->bhw", py, px)  # (B,H,W)
    P = P / P.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return F.mse_loss(P.unsqueeze(1), Apos)


def recon_l1(px: torch.Tensor, py: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error (L1) between predicted separable map and normalized attention.
    Returns a scalar (lower is better).
    """
    Apos = A.clamp_min(0)
    Apos = Apos / Apos.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    P = torch.einsum("bh,bw->bhw", py, px)  # (B,H,W)
    P = P / P.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return F.l1_loss(P.unsqueeze(1), Apos)


def build_axis_cdf_targets(A: torch.Tensor, L: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A: (B,1,H,W) >=0  ->  Fx,Fy in (B,L), monotone, end at 1.0
    """
    A = torch.nan_to_num(A.float().clamp_min(0), nan=0.0, posinf=0.0, neginf=0.0)
    mx = A.sum(dim=2).squeeze(1)  # (B, W)
    my = A.sum(dim=3).squeeze(1)  # (B, H)
    mx = mx / mx.sum(dim=1, keepdim=True).clamp_min(1e-6)
    my = my / my.sum(dim=1, keepdim=True).clamp_min(1e-6)
    mx_ds = F.adaptive_avg_pool1d(mx.unsqueeze(1), L).squeeze(1)
    my_ds = F.adaptive_avg_pool1d(my.unsqueeze(1), L).squeeze(1)
    Fx = torch.cumsum(mx_ds, dim=1)
    Fy = torch.cumsum(my_ds, dim=1)
    Fx = Fx / Fx[:, -1:].clamp_min(1e-6)
    Fx[:, -1] = 1.0
    Fy = Fy / Fy[:, -1:].clamp_min(1e-6)
    Fy[:, -1] = 1.0
    return Fx, Fy


def cdf_from_density_downsample(p: torch.Tensor, L: int) -> torch.Tensor:
    """Convert densities p:(B,N) -> CDF:(B,L) via pool(pdf)->cumsum->renorm."""
    p = torch.nan_to_num(p.float().clamp_min(0), nan=0.0, posinf=0.0, neginf=0.0)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-6)
    p_ds = F.adaptive_avg_pool1d(p.unsqueeze(1), L).squeeze(1)
    Fp = torch.cumsum(p_ds, dim=1)
    Fp = Fp / Fp[:, -1:].clamp_min(1e-6)
    B = Fp.size(0)
    Fp = torch.cat([Fp[:, :-1], Fp.new_ones(B, 1)], dim=1)
    return Fp


