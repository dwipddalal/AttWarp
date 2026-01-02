from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import FullDataset
from LLava_loader import LLaVAFeatHelper
from .checkpoint_utils import (
    cdf_from_density,
    gt_marginals,
    resample_cdf,
    warp_from_cdf_torch,
    upsample_pdf_right_inverse,
    plot_axis_pdf_overlay,
    plot_axis_cdf_overlay,
)

from .config import MarginalNetFullDatasetConfig
from .data import collate_str, load_gqa_transform_map, transform_for_sample
from .experiment import create_experiment_run_dir
from .losses import recon_l1
from .model import MarginalNet, entropy, mix_with_uniform
from .plots import plot_train_val_curves
from .wandb_utils import init_wandb


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_loop(cfg: MarginalNetFullDatasetConfig | None = None) -> Path:
    """
    Train MarginalNet on FullDataset (GQA/TextVQA/DocVQA). Returns run_dir.
    """
    cfg = cfg or MarginalNetFullDatasetConfig()

    # determinism
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    _seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")

    run_dir = create_experiment_run_dir(
        experiments_root=cfg.experiments_root,
        project_root=cfg.llava_repo_root,  # repo root is project root here
    )
    save_dir = run_dir / "checkpoints"

    # dataset & loader (logs-based FullDataset)
    full_ds = FullDataset(
        gqa_root_dir=cfg.gqa_root_dir,
        textvqa_root_dir=cfg.textvqa_root_dir,
        docvqa_root_dir=cfg.docvqa_root_dir,
        num_samples_per_dataset=cfg.num_per_ds,
        artifact_type="relative",
        artifact_layer=16,
        random_seed=cfg.seed,
        image_size=cfg.image_size,
    )
    train_len = int(0.9 * len(full_ds))
    val_len = len(full_ds) - train_len
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_str,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_str,
        drop_last=False,
    )

    # frozen LLaVA features
    feats = LLaVAFeatHelper(repo_root=cfg.llava_repo_root, model_path=cfg.llava_model, device=device)

    # probe dims once
    with torch.no_grad():
        b0 = next(iter(loader))
        img0 = b0["image"][:1].to(device)
        txt0 = [b0["q_str"][0]]
        fmap0 = feats.visual_tokens(img0)  # (1, Dv, hv, wv)
        ttok0, _tmask0 = feats.text_tokens(txt0)
        Dv = fmap0.size(1)
        Dt = ttok0.size(-1)

    net = MarginalNet(d_vis_in=Dv, d_txt_in=Dt, hidden=cfg.hidden, eps=cfg.eps).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    run = init_wandb(
        run_dir=run_dir,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        cfg={
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "wd": cfg.wd,
            "grad_clip": cfg.grad_clip,
            "workers": cfg.workers,
            "image_size": cfg.image_size,
            "num_per_ds": cfg.num_per_ds,
            "hidden": cfg.hidden,
            "w_cdf": cfg.w_cdf,
            "axis_len": cfg.axis_len,
            "llava_model": cfg.llava_model,
            "gqa_root_dir": cfg.gqa_root_dir,
            "textvqa_root_dir": cfg.textvqa_root_dir,
            "docvqa_root_dir": cfg.docvqa_root_dir,
            "artifact_type": "relative",
            "artifact_layer": 16,
            "vis_every": cfg.vis_every,
        },
    )

    # watch is best-effort
    try:
        if not os.environ.get("WANDB_DISABLED"):
            import wandb

            wandb.watch(net, log="gradients", log_freq=100, log_graph=False)
    except Exception:
        pass

    global_step = 0
    train_curve: List[float] = []
    val_curve: List[float] = []

    # mapping for GQA bucket -> transform
    gqa_transform_map = load_gqa_transform_map("")

    for epoch in range(cfg.epochs):
        net.train()
        train_loss_sum = 0.0
        train_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for batch in pbar:
            img = batch["image"].to(device)  # (B,3,H,W)
            att = batch["attention_map"].to(device)  # (B,1,H,W)
            A_full = att

            # dataset-specific transform on full-res
            ds_names = batch.get("dataset", None)
            buckets = batch.get("bucket", None)
            if ds_names is not None:
                B_att = A_full.size(0)
                tfms: List[str] = [
                    transform_for_sample(
                        ds_names[i] if i < len(ds_names) else None,
                        buckets[i] if (buckets is not None and i < len(buckets)) else None,
                        gqa_transform_map,
                    )
                    for i in range(B_att)
                ]
                with torch.no_grad():
                    mask_sqrt = torch.tensor(
                        [1.0 if t == "sqrt" else 0.0 for t in tfms],
                        device=device,
                        dtype=A_full.dtype,
                    ).view(B_att, 1, 1, 1)
                A_full_pos = A_full.clamp_min(0.0)
                A_full_sqrt = A_full_pos.sqrt()
                A_full = A_full_sqrt * mask_sqrt + A_full_pos * (1.0 - mask_sqrt)

            # pool to low-res after transform
            A = F.adaptive_avg_pool2d(A_full, (24, 24))
            qs = batch["q_str"]

            # sanitize
            img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
            B, _, H, W = A.shape

            with torch.no_grad():
                fmap_v = feats.visual_tokens(img)
                ttok, tmask = feats.text_tokens(qs)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                px_pred, py_pred = net(fmap_v, H, W, ttok, tmask)

                alpha = max(cfg.alpha0 * (1.0 - global_step / cfg.alpha_decay_steps), 0.0)
                px_s = mix_with_uniform(px_pred, alpha)
                py_s = mix_with_uniform(py_pred, alpha)

                # upsample PDFs to image resolution first, then compare
                px_img = upsample_pdf_right_inverse(px_s, img.size(-1)).clamp_min(0)
                py_img = upsample_pdf_right_inverse(py_s, img.size(-2)).clamp_min(0)

                px_gt_vis, py_gt_vis = gt_marginals(A)

                # override GT to uniform (no warping) for GQA samples with transform 'none'
                if ds_names is not None:
                    B_att2 = A_full.size(0)
                    tfms2: List[str] = [
                        transform_for_sample(
                            ds_names[i] if i < len(ds_names) else None,
                            buckets[i] if (buckets is not None and i < len(buckets)) else None,
                            gqa_transform_map,
                        )
                        for i in range(B_att2)
                    ]
                    none_mask = torch.tensor([True if t == "none" else False for t in tfms2], device=A.device)
                    if none_mask.any():
                        Lx = px_gt_vis.size(1)
                        Ly = py_gt_vis.size(1)
                        px_gt_vis[none_mask] = 1.0 / max(1, Lx)
                        py_gt_vis[none_mask] = 1.0 / max(1, Ly)

                px_gt_img = upsample_pdf_right_inverse(px_gt_vis, img.size(-1)).clamp_min(0)
                py_gt_img = upsample_pdf_right_inverse(py_gt_vis, img.size(-2)).clamp_min(0)

                # normalize PDFs to be safe
                px_img = px_img / px_img.sum(dim=1, keepdim=True).clamp_min(1e-6)
                py_img = py_img / py_img.sum(dim=1, keepdim=True).clamp_min(1e-6)
                px_gt_img = px_gt_img / px_gt_img.sum(dim=1, keepdim=True).clamp_min(1e-6)
                py_gt_img = py_gt_img / py_gt_img.sum(dim=1, keepdim=True).clamp_min(1e-6)

                # Image-resolution PDF L1 loss
                L_pdf = F.l1_loss(px_img, px_gt_img) + F.l1_loss(py_img, py_gt_img)

                L_kl = torch.tensor(0.0, device=device)
                L_rec = torch.tensor(0.0, device=device)
                L_ent = cfg.ent_weight * (entropy(px_s) + entropy(py_s))

                loss = cfg.w_cdf * L_pdf  # - L_ent
                train_l1 = recon_l1(px_s, py_s, A)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            train_loss_sum += float(loss.item())
            train_batches += 1

            if (global_step % cfg.log_every) == 0:
                run.log(
                    {
                        "global_step": global_step,
                        "loss/train_total": float(loss.item()),
                        "loss/train_pdf": float(L_pdf.item()),
                        "loss/train_recon_l1": float(train_l1.item()),
                        "lr": float(opt.param_groups[0]["lr"]),
                        "grad_norm": float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
                    }
                )

            # occasional visual debug
            if (global_step % cfg.vis_every) == 0:
                with torch.no_grad():
                    px_pred_img = upsample_pdf_right_inverse(px_pred, img.size(-1)).clamp_min(0)
                    py_pred_img = upsample_pdf_right_inverse(py_pred, img.size(-2)).clamp_min(0)
                    Fx_img = cdf_from_density(px_pred_img)
                    Fy_img = cdf_from_density(py_pred_img)
                    Wimg = warp_from_cdf_torch(img, Fx_img, Fy_img).clamp(0, 1)

                    px_gt_vis2, py_gt_vis2 = gt_marginals(A)
                    px_gt_img2 = upsample_pdf_right_inverse(px_gt_vis2, img.size(-1)).clamp_min(0)
                    py_gt_img2 = upsample_pdf_right_inverse(py_gt_vis2, img.size(-2)).clamp_min(0)
                    Fx_gt_img = cdf_from_density(px_gt_img2)
                    Fy_gt_img = cdf_from_density(py_gt_img2)
                    Wimg_gt = warp_from_cdf_torch(img, Fx_gt_img, Fy_gt_img).clamp(0, 1)

                    import torchvision.utils as vutils
                    from PIL import Image

                    heat = A / (A.amax(dim=(2, 3), keepdim=True) + 1e-6)
                    heat = heat.repeat(1, 3, 1, 1)
                    heat = F.interpolate(heat, size=img.shape[-2:], mode="bilinear", align_corners=False)

                    num_vis = min(4, img.size(0))
                    grid = torch.cat([img[:num_vis], heat[:num_vis], Wimg[:num_vis], Wimg_gt[:num_vis]], dim=0)
                    grid = vutils.make_grid((grid * 255).byte(), nrow=num_vis)
                    out_dir = run_dir / "debug"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    grid_path = out_dir / f"dbg_{global_step:06d}.png"
                    Image.fromarray(grid.permute(1, 2, 0).cpu().numpy()).save(grid_path)

                    warped_dir = out_dir / "warped"
                    warped_dir.mkdir(parents=True, exist_ok=True)
                    meta = {"global_step": int(global_step), "grid_image_path": str(grid_path), "samples": []}

                    datasets = batch.get("dataset", [None] * num_vis)
                    buckets_dbg = batch.get("bucket", [None] * num_vis)
                    for i in range(num_vis):
                        w_img = (Wimg[i].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                        w_path = warped_dir / f"dbg_{global_step:06d}_sample{i}.png"
                        Image.fromarray(w_img).save(w_path)
                        w_img_gt = (Wimg_gt[i].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                        w_gt_path = warped_dir / f"dbg_{global_step:06d}_sample{i}_gt.png"
                        Image.fromarray(w_img_gt).save(w_gt_path)
                        meta["samples"].append(
                            {
                                "index": int(i),
                                "dataset": str(datasets[i]) if datasets and i < len(datasets) else None,
                                "question": str(qs[i]) if qs and i < len(qs) else None,
                                "bucket": str(buckets_dbg[i]) if (buckets_dbg and i < len(buckets_dbg)) else None,
                                "warped_image_path": str(w_path),
                                "gt_warped_image_path": str(w_gt_path),
                            }
                        )
                    with open(out_dir / f"dbg_{global_step:06d}.json", "w") as f:
                        json.dump(meta, f, indent=2)

                    # PDF/CDF plots
                    pdf_dir = out_dir / "pdfs"
                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    for i in range(num_vis):
                        pdf_path = pdf_dir / f"train_{global_step:06d}_sample{i}.png"
                        plot_axis_pdf_overlay(
                            px_pred[i],
                            py_pred[i],
                            px_gt_vis2[i],
                            py_gt_vis2[i],
                            pdf_path,
                            title_prefix="Train",
                            sample_idx=i,
                        )

                    cdf_dir = out_dir / "cdfs"
                    cdf_dir.mkdir(parents=True, exist_ok=True)

                    px_gt_full, py_gt_full = gt_marginals(A_full)
                    Fx_gt_full = cdf_from_density(px_gt_full)
                    Fy_gt_full = cdf_from_density(py_gt_full)
                    Fx_gt_full_img = resample_cdf(Fx_gt_full, img.size(-1))
                    Fy_gt_full_img = resample_cdf(Fy_gt_full, img.size(-2))

                    for i in range(num_vis):
                        cdf_path = cdf_dir / f"train_{global_step:06d}_sample{i}.png"
                        plot_axis_cdf_overlay(
                            Fx_pred=Fx_img[i],
                            Fy_pred=Fy_img[i],
                            Fx_gt=Fx_gt_img[i],
                            Fy_gt=Fy_gt_img[i],
                            out_path=cdf_path,
                            title_prefix="Train (image-res CDF)",
                            sample_idx=i,
                            Fx_gt_full=Fx_gt_full_img[i],
                            Fy_gt_full=Fy_gt_full_img[i],
                        )

                    # W&B image logging (best-effort)
                    can_log = hasattr(run, "log") and not os.environ.get("WANDB_DISABLED")
                    if can_log:
                        try:
                            import wandb

                            imgs = {"debug/grid": wandb.Image(str(grid_path), caption=f"step {global_step}")}
                            for i in range(num_vis):
                                pred_path = warped_dir / f"dbg_{global_step:06d}_sample{i}.png"
                                gt_path = warped_dir / f"dbg_{global_step:06d}_sample{i}_gt.png"
                                p = pdf_dir / f"train_{global_step:06d}_sample{i}.png"
                                c = cdf_dir / f"train_{global_step:06d}_sample{i}.png"
                                if p.exists():
                                    imgs[f"pdf/train_sample{i}"] = wandb.Image(str(p))
                                if pred_path.exists():
                                    imgs[f"debug/pred_sample{i}"] = wandb.Image(str(pred_path))
                                if gt_path.exists():
                                    imgs[f"debug/gt_sample{i}"] = wandb.Image(str(gt_path))
                                if c.exists():
                                    imgs[f"cdf/train_sample{i}"] = wandb.Image(str(c))
                            run.log(imgs, step=global_step)
                        except Exception:
                            pass

        # validation each epoch
        net.eval()
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            did_val_debug = False
            for vbatch in val_loader:
                img = vbatch["image"].to(device)
                v_att = vbatch["attention_map"].to(device)

                v_ds_names = vbatch.get("dataset", None)
                v_buckets = vbatch.get("bucket", None)
                if v_ds_names is not None:
                    B_vatt = v_att.size(0)
                    v_tfms: List[str] = [
                        transform_for_sample(
                            v_ds_names[i] if i < len(v_ds_names) else None,
                            v_buckets[i] if (v_buckets is not None and i < len(v_buckets)) else None,
                            gqa_transform_map,
                        )
                        for i in range(B_vatt)
                    ]
                    v_mask_sqrt = torch.tensor(
                        [1.0 if t == "sqrt" else 0.0 for t in v_tfms],
                        device=device,
                        dtype=v_att.dtype,
                    ).view(B_vatt, 1, 1, 1)
                    v_att_pos = v_att.clamp_min(0.0)
                    v_att_sqrt = v_att_pos.sqrt()
                    v_att = v_att_sqrt * v_mask_sqrt + v_att_pos * (1.0 - v_mask_sqrt)

                A = F.adaptive_avg_pool2d(v_att, (24, 24))
                qs = vbatch["q_str"]
                fmap_v = feats.visual_tokens(img)
                ttok, tmask = feats.text_tokens(qs)
                px_pred, py_pred = net(fmap_v, A.size(2), A.size(3), ttok, tmask)

                px_img = upsample_pdf_right_inverse(px_pred, img.size(-1)).clamp_min(0)
                py_img = upsample_pdf_right_inverse(py_pred, img.size(-2)).clamp_min(0)
                px_gt_vis, py_gt_vis = gt_marginals(A)
                px_gt_img = upsample_pdf_right_inverse(px_gt_vis, img.size(-1)).clamp_min(0)
                py_gt_img = upsample_pdf_right_inverse(py_gt_vis, img.size(-2)).clamp_min(0)

                px_img = px_img / px_img.sum(dim=1, keepdim=True).clamp_min(1e-6)
                py_img = py_img / py_img.sum(dim=1, keepdim=True).clamp_min(1e-6)
                px_gt_img = px_gt_img / px_gt_img.sum(dim=1, keepdim=True).clamp_min(1e-6)
                py_gt_img = py_gt_img / py_gt_img.sum(dim=1, keepdim=True).clamp_min(1e-6)

                val_pdf = F.l1_loss(px_img, px_gt_img) + F.l1_loss(py_img, py_gt_img)
                val_loss_sum += float((cfg.w_cdf * val_pdf).item())
                val_batches += 1

                # save validation debug visualizations once per epoch (match original script)
                if not did_val_debug:
                    with torch.no_grad():
                        px_pred_img = upsample_pdf_right_inverse(px_pred, img.size(-1)).clamp_min(0)
                        py_pred_img = upsample_pdf_right_inverse(py_pred, img.size(-2)).clamp_min(0)
                        Fx_img = cdf_from_density(px_pred_img)
                        Fy_img = cdf_from_density(py_pred_img)

                        Wimg = warp_from_cdf_torch(img, Fx_img, Fy_img).clamp(0, 1)

                        # ground-truth CDF warp from attention-derived marginals (apply per-sample transform)
                        A_vis_ds = F.adaptive_avg_pool2d(v_att, (24, 24))
                        px_gt_vis_dbg, py_gt_vis_dbg = gt_marginals(A_vis_ds)
                        if v_ds_names is not None:
                            v_none_mask = torch.tensor([True if t == "none" else False for t in v_tfms], device=A_vis_ds.device)
                            if v_none_mask.any():
                                Lx = px_gt_vis_dbg.size(1)
                                Ly = py_gt_vis_dbg.size(1)
                                px_gt_vis_dbg[v_none_mask] = 1.0 / max(1, Lx)
                                py_gt_vis_dbg[v_none_mask] = 1.0 / max(1, Ly)

                        px_gt_img_dbg = upsample_pdf_right_inverse(px_gt_vis_dbg, img.size(-1)).clamp_min(0)
                        py_gt_img_dbg = upsample_pdf_right_inverse(py_gt_vis_dbg, img.size(-2)).clamp_min(0)
                        Fx_gt_img = cdf_from_density(px_gt_img_dbg)
                        Fy_gt_img = cdf_from_density(py_gt_img_dbg)
                        Wimg_gt = warp_from_cdf_torch(img, Fx_gt_img, Fy_gt_img).clamp(0, 1)

                        import torchvision.utils as vutils

                        # attention heat (use VIS downsampled attention for consistency)
                        heat = A_vis_ds / (A_vis_ds.amax(dim=(2, 3), keepdim=True) + 1e-6)
                        heat = heat.repeat(1, 3, 1, 1)
                        heat = F.interpolate(heat, size=img.shape[-2:], mode="bilinear", align_corners=False)

                        num_vis = min(4, img.size(0))
                        grid = torch.cat([img[:num_vis], heat[:num_vis], Wimg[:num_vis], Wimg_gt[:num_vis]], dim=0)
                        grid = vutils.make_grid((grid * 255).byte(), nrow=num_vis)

                        from PIL import Image

                        out_dir = run_dir / "debug_validation"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        grid_path = out_dir / f"val_{epoch + 1:03d}_{global_step:06d}.png"
                        Image.fromarray(grid.permute(1, 2, 0).cpu().numpy()).save(grid_path)

                        # Also compute ORIGINAL GT warp directly from the raw validation attention map
                        A_full_orig = vbatch["attention_map"].to(device)
                        px_gt_full_orig, py_gt_full_orig = gt_marginals(A_full_orig)
                        Fx_gt_full_orig = cdf_from_density(px_gt_full_orig)
                        Fy_gt_full_orig = cdf_from_density(py_gt_full_orig)
                        Wimg_gt_orig = warp_from_cdf_torch(img, Fx_gt_full_orig, Fy_gt_full_orig)

                        # Build 5-row grid: [orig | attention | pred-warp | ds-GT-warp | original-GT-warp]
                        grid = torch.cat(
                            [
                                img[:num_vis],
                                heat[:num_vis],
                                Wimg[:num_vis],
                                Wimg_gt[:num_vis],
                                Wimg_gt_orig[:num_vis],
                            ],
                            dim=0,
                        )
                        grid = vutils.make_grid((grid * 255).byte(), nrow=num_vis)

                        from PIL import ImageDraw, ImageFont
                        import textwrap

                        grid_img = Image.fromarray(grid.permute(1, 2, 0).cpu().numpy())

                        # Create a side panel to display questions per sample
                        panel_w = 400
                        panel = Image.new("RGB", (panel_w, grid_img.height), (0, 0, 0))
                        draw = ImageDraw.Draw(panel)
                        # Fonts: enlarge question font 3x
                        try:
                            font_title = ImageFont.truetype("DejaVuSans.ttf", size=24)
                            font_label = ImageFont.truetype("DejaVuSans.ttf", size=22)
                            font_q = ImageFont.truetype("DejaVuSans.ttf", size=36)
                        except Exception:
                            font_title = ImageFont.load_default()
                            font_label = ImageFont.load_default()
                            font_q = ImageFont.load_default()

                        y = 10
                        draw.text((10, y), f"Epoch {epoch + 1} • Step {global_step}", fill=(255, 255, 255), font=font_title)
                        y += (font_title.getbbox("Ag")[3] - font_title.getbbox("Ag")[1]) + 4
                        draw.text((10, y), "Questions:", fill=(255, 255, 0), font=font_label)
                        y += (font_label.getbbox("Ag")[3] - font_label.getbbox("Ag")[1]) + 6
                        for i in range(num_vis):
                            q_text = str(qs[i]) if qs and i < len(qs) else ""
                            wrapped = textwrap.wrap(q_text, width=24)
                            draw.text((10, y), f"{i}:", fill=(180, 220, 255), font=font_label)
                            y += (font_label.getbbox("Ag")[3] - font_label.getbbox("Ag")[1]) + 2
                            for line in wrapped:
                                draw.text((24, y), line, fill=(220, 220, 220), font=font_q)
                                y += (font_q.getbbox(line)[3] - font_q.getbbox(line)[1]) + 4
                            y += 8

                        combined = Image.new("RGB", (grid_img.width + panel_w, grid_img.height), (0, 0, 0))
                        combined.paste(grid_img, (0, 0))
                        combined.paste(panel, (grid_img.width, 0))
                        combined.save(grid_path)

                        # --- PDF overlay plots (VAL) ---
                        pdf_dir = out_dir / "pdfs"
                        pdf_dir.mkdir(parents=True, exist_ok=True)
                        for i in range(num_vis):
                            pdf_path = pdf_dir / f"val_{epoch + 1:03d}_{global_step:06d}_sample{i}.png"
                            plot_axis_pdf_overlay(
                                px_pred[i],
                                py_pred[i],
                                px_gt_vis_dbg[i],
                                py_gt_vis_dbg[i],
                                pdf_path,
                                title_prefix=f"Val (epoch {epoch + 1})",
                                sample_idx=i,
                            )

                        cdf_dir = out_dir / "cdfs"
                        cdf_dir.mkdir(parents=True, exist_ok=True)

                        # full-resolution GT CDFs from validation attention map (per-sample transforms already applied)
                        A_full_val = v_att
                        px_gt_full, py_gt_full = gt_marginals(A_full_val)
                        if v_ds_names is not None:
                            v_none_mask_full = torch.tensor(
                                [True if t == "none" else False for t in v_tfms],
                                device=A_full_val.device,
                            )
                            if v_none_mask_full.any():
                                Lx = px_gt_full.size(1)
                                Ly = py_gt_full.size(1)
                                px_gt_full[v_none_mask_full] = 1.0 / max(1, Lx)
                                py_gt_full[v_none_mask_full] = 1.0 / max(1, Ly)

                        Fx_gt_full = cdf_from_density(px_gt_full)
                        Fy_gt_full = cdf_from_density(py_gt_full)
                        Fx_gt_full_img = resample_cdf(Fx_gt_full, img.size(-1))
                        Fy_gt_full_img = resample_cdf(Fy_gt_full, img.size(-2))

                        for i in range(num_vis):
                            cdf_path = cdf_dir / f"val_{epoch + 1:03d}_{global_step:06d}_sample{i}.png"
                            plot_axis_cdf_overlay(
                                Fx_pred=Fx_img[i],
                                Fy_pred=Fy_img[i],
                                Fx_gt=Fx_gt_img[i],
                                Fy_gt=Fy_gt_img[i],
                                out_path=cdf_path,
                                title_prefix=f"Val (epoch {epoch + 1}) image-res CDF",
                                sample_idx=i,
                                Fx_gt_full=Fx_gt_full_img[i],
                                Fy_gt_full=Fy_gt_full_img[i],
                            )

                        # log to W&B
                        can_log = hasattr(run, "log") and not os.environ.get("WANDB_DISABLED")
                        if can_log:
                            try:
                                import wandb

                                imgs = {
                                    "debug_val/grid": wandb.Image(
                                        str(grid_path), caption=f"epoch {epoch + 1} step {global_step}"
                                    ),
                                }
                                for i in range(num_vis):
                                    p = pdf_dir / f"val_{epoch + 1:03d}_{global_step:06d}_sample{i}.png"
                                    c = cdf_dir / f"val_{epoch + 1:03d}_{global_step:06d}_sample{i}.png"
                                    if p.exists():
                                        imgs[f"pdf_val/sample{i}"] = wandb.Image(str(p))
                                    if c.exists():
                                        imgs[f"cdf_val/sample{i}"] = wandb.Image(str(c))
                                run.log(imgs, step=global_step)
                            except Exception:
                                pass

                    did_val_debug = True

        val_loss = val_loss_sum / max(val_batches, 1)
        train_loss_mean = train_loss_sum / max(train_batches, 1)
        print(f"Epoch {epoch + 1}: train_loss={train_loss_mean:.4f} | val_loss={val_loss:.4f}")

        train_curve.append(train_loss_mean)
        val_curve.append(val_loss)

        run.log(
            {
                "epoch": epoch + 1,
                "loss/train_epoch": train_loss_mean,
                "loss/val_epoch": val_loss,
            },
            step=global_step,
        )

        try:
            plot_train_val_curves(
                train_curve,
                val_curve,
                run_dir / "curves" / "loss.png",
                title="CDF L1 Loss",
                ylabel="Loss",
            )
        except Exception:
            pass

        ckpt_path = save_dir / f"marginal_net_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model": net.state_dict(),
                "opt": opt.state_dict(),
                "cfg": {
                    "seed": cfg.seed,
                    "epochs": cfg.epochs,
                    "batch_size": cfg.batch_size,
                    "lr": cfg.lr,
                    "wd": cfg.wd,
                    "grad_clip": cfg.grad_clip,
                    "workers": cfg.workers,
                    "image_size": cfg.image_size,
                    "num_per_ds": cfg.num_per_ds,
                    "hidden": cfg.hidden,
                    "w_cdf": cfg.w_cdf,
                    "axis_len": cfg.axis_len,
                    "llava_model": cfg.llava_model,
                },
            },
            ckpt_path,
        )

        # artifact logging (best-effort)
        try:
            if not os.environ.get("WANDB_DISABLED"):
                import wandb

                art = wandb.Artifact(f"marginal_net_epoch_{epoch + 1}", type="checkpoint")
                art.add_file(str(ckpt_path))
                run.log_artifact(art)
        except Exception:
            pass

    print("Training complete. Saved to:", str(run_dir))
    try:
        run.finish()
    except Exception:
        pass
    return run_dir


