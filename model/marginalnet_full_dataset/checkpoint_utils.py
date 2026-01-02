import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import sys
import pdb

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def _make_strictly_increasing(Fcdf: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    Fcdf = torch.nan_to_num(Fcdf, nan=0.0, posinf=1.0, neginf=0.0)
    Fcdf_nd, _ = torch.cummax(Fcdf, dim=1)      # non-decreasing
    B, N = Fcdf_nd.shape
    min_step = eps / max(N, 1)
    d = Fcdf_nd[:, 1:] - Fcdf_nd[:, :-1]
    d = torch.clamp(d, min=min_step)
    Fcdf_fix = torch.cat([Fcdf_nd[:, :1], Fcdf_nd[:, :1] + torch.cumsum(d, dim=1)], dim=1)
    last = Fcdf_fix[:, -1:].clamp_min(1e-6)
    Fcdf_fix = (Fcdf_fix / last).clamp(0.0, 1.0)
    Fcdf_fix[:, -1] = 1.0
    return Fcdf_fix

def cdf_from_density(p: torch.Tensor) -> torch.Tensor:
    """p: (B,N) -> (B,N) non-decreasing CDF in [0,1], ends at 1.

    Robust to NaNs/Infs and negative densities: clamps to nonnegative and
    renormalizes before cumulative sum to guarantee monotonic CDF.
    """
    p = torch.nan_to_num(p.float().clamp_min(0), nan=0.0, posinf=0.0, neginf=0.0)
    denom = p.sum(dim=1, keepdim=True).clamp_min(1e-6)
    p = p / denom
    Fp = torch.cumsum(p, dim=1)
    Fp[:, -1] = 1.0
    return Fp

def gt_marginals(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """A:(B,1,H,W) -> (px_gt:(B,W), py_gt:(B,H)) normalized."""
    B, _, H, W = A.shape
    Apos = A.clamp_min(0)
    mx = Apos.sum(dim=2).squeeze(1)  # (B,W)
    my = Apos.sum(dim=3).squeeze(1)  # (B,H)
    mx = mx / mx.sum(dim=1, keepdim=True).clamp_min(1e-6)
    my = my / my.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return mx, my

def resample_cdf(Fcdf: torch.Tensor, target_len: int) -> torch.Tensor:
    # Fcdf: (B, N) non-decreasing in [0,1]
    Fcdf = _make_strictly_increasing(Fcdf.float())
    # use linear 1D interpolation along the last dim
    F_up = F.interpolate(
        Fcdf.unsqueeze(1), size=target_len, mode='linear', align_corners=True
    ).squeeze(1)
    # enforce monotonicity and [0,1]
    F_up = _make_strictly_increasing(F_up)
    return F_up

def upsample_pdf_right_inverse(y: torch.Tensor, target_len: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Right-inverse upsampling for AdaptiveAvgPool1d.

    Given a downsampled 1D signal y (PDF) of length L_out produced by
    adaptive average pooling from some length L_in (=target_len), construct
    x_hat of length L_in such that:

        adaptive_avg_pool1d(x_hat, L_out) == y   (up to FP tolerance)

    This is the minimum-norm right-inverse x_hat = A^T (A A^T)^{-1} y,
    where A encodes the pooling windows. Works for batched inputs.

    Args:
        y:  (..., L_out) tensor; supports (L_out,), (B, L_out), or (B, C, L_out)
        target_len: L_in, the desired upsampled length
        eps: small Tikhonov diagonal added to A A^T for numerical stability

    Returns:
        x_hat with shape matching y but last-dim = target_len
    """
    if y.dim() == 1:
        yN = y.unsqueeze(0)  # (1, L_out)
        batch_mul = (1,)
    elif y.dim() == 2:
        yN = y  # (B, L_out)
        batch_mul = (y.size(0),)
    elif y.dim() == 3:
        B, C, L_out = y.size()
        yN = y.reshape(B * C, L_out)  # (N, L_out)
        batch_mul = (B, C)
    else:
        raise ValueError(f"upsample_pdf_right_inverse expects 1D/2D/3D y; got shape {tuple(y.shape)}")

    L_out = yN.size(1)
    L_in = int(target_len)
    device = y.device
    dtype = y.dtype

    # Compute pooling window boundaries matching PyTorch's AdaptiveAvgPool1d
    i = torch.arange(L_out, device=device)
    starts = (i * L_in // L_out).long()
    ends = (((i + 1) * L_in + L_out - 1) // L_out).long()  # integer ceil

    # Build pooling matrix A (L_out x L_in)
    A = torch.zeros(L_out, L_in, device=device, dtype=dtype)
    for k in range(L_out):
        s = int(starts[k].item())
        e = int(ends[k].item())
        A[k, s:e] = 1.0 / max(e - s, 1)

    # Solve x_hat = A^T (A A^T + eps I)^{-1} y  without explicit inverse
    AAT = A @ A.T
    if eps > 0:
        AAT = AAT + eps * torch.eye(L_out, device=device, dtype=dtype)
    rhs = yN.T  # (L_out, N)
    tmp = torch.linalg.solve(AAT, rhs)  # (L_out, N)
    x_hat = (A.T @ tmp).T  # (N, L_in)

    # Restore original batch/channel shape
    if y.dim() == 1:
        x_hat = x_hat.squeeze(0)  # (L_in,)
    elif y.dim() == 2:
        pass  # (B, L_in)
    else:
        x_hat = x_hat.reshape(batch_mul[0], batch_mul[1], L_in)

    return x_hat

def warp_from_cdf_torch(img: torch.Tensor,
                        Fx_img: torch.Tensor,  # (B, W) CDF in [0,1]
                        Fy_img: torch.Tensor,  # (B, H) CDF in [0,1]
                        out_size: tuple | None = None) -> torch.Tensor:
    """
    EXACT same mapping as your numpy snippet, but batched and safe.
    Uses cv2.remap per-sample (INTER_LINEAR + BORDER_REPLICATE).

    img:    (B,C,H,W)  uint8 or float
    Fx_img: (B,W)      CDF along X at image resolution, values in [0,1]
    Fy_img: (B,H)      CDF along Y at image resolution, values in [0,1]
    out_size: (H_out, W_out) or None -> keep (H,W)
    """
    assert img.ndim == 4, f"img must be (B,C,H,W); got {img.shape}"
    B, C, H, W = img.shape
    H_out, W_out = (H, W) if out_size is None else out_size

    # move to CPU/NumPy
    to_cpu = lambda t: t.detach().cpu().numpy()
    imgs_np = img.detach().cpu().permute(0, 2, 3, 1).numpy()  # (B,H,W,C)
    Fx_np   = to_cpu(Fx_img)
    Fy_np   = to_cpu(Fy_img)

    out_list = []
    for b in range(B):
        Fx_b = np.asarray(Fx_np[b], dtype=np.float32).reshape(-1)  # (W,)
        Fy_b = np.asarray(Fy_np[b], dtype=np.float32).reshape(-1)  # (H,)

        # sanity checks prevent the exact error you saw
        if Fx_b.shape[0] != W:
            raise ValueError(f"Fx_img[{b}] length {Fx_b.shape[0]} != image width W={W}")
        if Fy_b.shape[0] != H:
            raise ValueError(f"Fy_img[{b}] length {Fy_b.shape[0]} != image height H={H}")

        # ---- YOUR EXACT FORWARD/INVERSE CODE (single-sample) ----
        x_orig_coords = np.arange(W, dtype=np.float32)
        y_orig_coords = np.arange(H, dtype=np.float32)

        x_new_map_fwd = np.concatenate(([0.0], Fx_b)) * float(W_out)
        x_orig_map_fwd = np.concatenate(([0.0], x_orig_coords + 1.0))

        y_new_map_fwd = np.concatenate(([0.0], Fy_b)) * float(H_out)
        y_orig_map_fwd = np.concatenate(([0.0], y_orig_coords + 1.0))

        x_new_map_fwd[-1] = W_out
        y_new_map_fwd[-1] = H_out

        # (Optional) break ties if any; keeps np.interp happy
        if np.any(np.diff(x_new_map_fwd) <= 0):
            x_new_map_fwd += (1e-4 / max(W_out, 1)) * np.arange(x_new_map_fwd.size, dtype=np.float32)
        if np.any(np.diff(y_new_map_fwd) <= 0):
            y_new_map_fwd += (1e-4 / max(H_out, 1)) * np.arange(y_new_map_fwd.size, dtype=np.float32)

        x_target_coords = np.arange(W_out, dtype=np.float32)
        y_target_coords = np.arange(H_out, dtype=np.float32)
        map_x_orig = np.interp(x_target_coords, x_new_map_fwd, x_orig_map_fwd)
        map_y_orig = np.interp(y_target_coords, y_new_map_fwd, y_orig_map_fwd)

        final_map_x, final_map_y = np.meshgrid(map_x_orig, map_y_orig, indexing="xy")
        final_map_x = final_map_x.astype(np.float32, copy=False)
        final_map_y = final_map_y.astype(np.float32, copy=False)

        warped_b = cv2.remap(
            imgs_np[b], final_map_x, final_map_y,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        out_list.append(warped_b)

    outs = np.stack(out_list, axis=0)  # (B,H_out,W_out,C)
    out_t = torch.from_numpy(outs).permute(0, 3, 1, 2)  # (B,C,H_out,W_out)
    out_t = out_t.to(device=img.device, dtype=img.dtype)
    return out_t

def plot_axis_cdf_overlay(Fx_pred: torch.Tensor, Fy_pred: torch.Tensor,
                          Fx_gt: torch.Tensor,   Fy_gt: torch.Tensor,
                          out_path: Path, title_prefix: str, sample_idx: int,
                          Fx_gt_full: torch.Tensor | None = None,
                          Fy_gt_full: torch.Tensor | None = None) -> None:
    """
    Overlay GT vs Pred CDFs for X and Y with shaded differences, residual subplots,
    and an auto-zoomed inset around the largest discrepancy. Saves a 2x2 figure.
    """
    try:
        eps = 1e-12

        def _to_np_cdf(t: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                arr = t.detach().cpu().float().view(-1).numpy()
            else:
                arr = np.asarray(t, dtype=np.float32).reshape(-1)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
            if arr.size == 0:
                return np.zeros((0,), dtype=np.float32)
            # enforce valid CDF: monotone, in [0,1], end at 1
            arr = np.maximum.accumulate(arr)
            last = float(arr[-1])
            if last <= 1e-12:
                arr = np.linspace(0.0, 1.0, len(arr), dtype=np.float32)
            else:
                arr /= last
            arr[-1] = 1.0
            return arr

        # tensors -> safe CDFs
        x_pred = _to_np_cdf(Fx_pred)
        x_gt   = _to_np_cdf(Fx_gt)
        y_pred = _to_np_cdf(Fy_pred)
        y_gt   = _to_np_cdf(Fy_gt)
        x_gt_full = _to_np_cdf(Fx_gt_full)
        y_gt_full = _to_np_cdf(Fy_gt_full)

        # metrics on CDFs
        def _cdf_metrics(pred: np.ndarray, gt: np.ndarray):
            diff = pred - gt
            adiff = np.abs(diff)
            ks = float(adiff.max())
            ks_i = int(adiff.argmax()) if adiff.size else 0
            mean_abs = float(adiff.mean())  # ≈ EMD for uniform bins
            rmse = float(np.sqrt(np.mean(diff**2)))
            return ks, ks_i, mean_abs, rmse, diff, adiff

        def _zoom_limits(adiff: np.ndarray, frac: float = 0.2):
            n = len(adiff)
            if n == 0: return 0, 0
            w = max(6, int(frac * n))
            c = np.convolve(adiff, np.ones(w, dtype=float), mode="valid")
            i0 = int(np.argmax(c))
            return i0, min(i0 + w, n - 1)

        # figure (2x2): top=overlays (X,Y), bottom=residuals (X,Y)
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        ax_x, ax_y = axes[0, 0], axes[0, 1]
        rx_x, rx_y = axes[1, 0], axes[1, 1]
        fig.suptitle(f"{title_prefix} • sample {sample_idx}", fontsize=13)

        def _overlay_one(ax_main, ax_resid, gt, pred, gt_full, axis_name: str):
            idx = np.arange(len(gt))
            ks, ks_i, mean_abs, rmse, diff, adiff = _cdf_metrics(pred, gt)

            # Overlay with steps + shaded |Δ|
            ax_main.step(idx, gt,   where="mid", linewidth=2, label=f"GT {axis_name}-cdf (downsampled)")
            ax_main.step(idx, pred, where="mid", linestyle="--", label=f"Pred {axis_name}-cdf")
            if gt_full is not None and len(gt_full) > 0:
                ax_main.step(np.arange(len(gt_full)), gt_full, where="mid", linestyle=":", label=f"GT {axis_name}-cdf (full)")
            ax_main.fill_between(idx, gt, pred, step="mid", alpha=0.25, label="|Δ|")

            ax_main.set_ylim(0, 1)
            ax_main.set_ylabel("CDF")
            ax_main.set_title(f"{axis_name}-axis • KS={ks:.4f}  mean|Δ|={mean_abs:.4f}  RMSE={rmse:.4f}")
            ax_main.grid(True, alpha=0.3)
            ax_main.legend(loc="best", fontsize=9)

            # Mark KS location
            if len(idx) > 0:
                ax_main.axvline(ks_i, linestyle=":", linewidth=1)

            # Auto-zoom inset around largest discrepancy
            i0, i1 = _zoom_limits(adiff)
            if i1 > i0:
                axins = inset_axes(ax_main, width="40%", height="45%", loc="lower right", borderpad=1.0)
                axins.step(idx, gt,   where="mid")
                axins.step(idx, pred, where="mid", linestyle="--")
                axins.fill_between(idx, gt, pred, step="mid", alpha=0.25)
                axins.set_xlim(i0, i1)
                local = np.r_[gt[i0:i1+1], pred[i0:i1+1]]
                lo, hi = local.min(), local.max()
                pad = 0.05 * max(hi - lo, eps)
                axins.set_ylim(max(0, lo - pad), min(1, hi + pad))
                mark_inset(ax_main, axins, loc1=2, loc2=4, fc="none", ec="0.5")

            # Residual subplot (signed ΔCDF)
            ax_resid.axhline(0.0, color="k", linewidth=1)
            ax_resid.step(idx, diff, where="mid", label="ΔCDF (Pred − GT)")
            ax_resid.fill_between(idx, 0.0, diff, step="mid", alpha=0.2)
            # Show ±KS as reference lines
            ax_resid.axhline( ks, linestyle=":", linewidth=1)
            ax_resid.axhline(-ks, linestyle=":", linewidth=1)
            # Robust symmetric limits
            lim = np.nanpercentile(np.abs(diff), 99.0) if diff.size else 0.01
            lim = max(lim, 1e-3)
            ax_resid.set_ylim(-1.15 * lim, 1.15 * lim)
            ax_resid.set_xlabel("bin")
            ax_resid.set_ylabel("ΔCDF")
            ax_resid.grid(True, alpha=0.3)
            # optional legend is usually not needed here:
            # ax_resid.legend(loc="best", fontsize=9)

        _overlay_one(ax_x, rx_x, x_gt, x_pred, x_gt_full, "X")
        _overlay_one(ax_y, rx_y, y_gt, y_pred, y_gt_full, "Y")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=200)
        plt.close(fig)
    except Exception:
        # Keep training robust even if plotting hiccups happen
        pass



def plot_axis_pdf_overlay(px_pred: torch.Tensor, py_pred: torch.Tensor,
                         px_gt: torch.Tensor, py_gt: torch.Tensor,
                         out_path: Path, title_prefix: str, sample_idx: int,
                         px_gt_full: torch.Tensor | None = None,
                         py_gt_full: torch.Tensor | None = None) -> None:
    """
    Save a 2-row matplotlib figure overlaying GT vs Pred PDFs for X and Y axes.
    px_*: (W,), py_*: (H,)
    """
    try:
        # make sure all series are valid + normalized for visualization
        def _to_np(t: torch.Tensor) -> np.ndarray:
            t = torch.nan_to_num(t.float().clamp_min(0), nan=0.0, posinf=0.0, neginf=0.0)
            t = t / t.sum().clamp_min(1e-6)
            return t.detach().cpu().numpy()

        x_pred = _to_np(px_pred)
        x_gt   = _to_np(px_gt)
        y_pred = _to_np(py_pred)
        y_gt   = _to_np(py_gt)
        x_gt_full = _to_np(px_gt_full) if px_gt_full is not None else None
        y_gt_full = _to_np(py_gt_full) if py_gt_full is not None else None

        fig = plt.figure(figsize=(8, 5.5))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(x_gt,  label="GT x-pdf (downsampled)",  linewidth=2)
        ax1.plot(x_pred, label="Pred x-pdf", linestyle="--")
        if x_gt_full is not None:
            ax1.plot(x_gt_full, label="GT x-pdf (full)", linestyle=":")
        ax1.set_ylabel("density")
        ax1.set_title(f"{title_prefix} • sample {sample_idx} • X-axis")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(y_gt,  label="GT y-pdf (downsampled)",  linewidth=2)
        ax2.plot(y_pred, label="Pred y-pdf", linestyle="--")
        if y_gt_full is not None:
            ax2.plot(y_gt_full, label="GT y-pdf (full)", linestyle=":")
        ax2.set_xlabel("bin")
        ax2.set_ylabel("density")
        ax2.set_title(f"{title_prefix} • sample {sample_idx} • Y-axis")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(str(out_path))
        plt.close(fig)
    except Exception:
        # keep training robust even if plotting hiccups happen
        pass

def _normalize_cdf_1d(F: np.ndarray) -> np.ndarray:
    """Make CDF safe: non-decreasing in [0,1] and ending at 1.0."""
    F = np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    F = np.maximum.accumulate(F)  # non-decreasing
    last = float(F[-1])
    if last <= 1e-12:  # degenerate -> fallback to uniform
        N = F.shape[0]
        F = np.linspace(0.0, 1.0, N, dtype=np.float32)
    else:
        F /= last
    F[-1] = 1.0
    return F

