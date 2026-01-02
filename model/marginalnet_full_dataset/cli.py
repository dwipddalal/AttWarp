from __future__ import annotations

import argparse
from typing import Sequence

from .config import MarginalNetFullDatasetConfig
from .trainer import train_loop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("MarginalNet training (FullDataset)")
    p.add_argument("--gqa-root-dir", type=str, default=MarginalNetFullDatasetConfig().gqa_root_dir)
    p.add_argument("--textvqa-root-dir", type=str, default=MarginalNetFullDatasetConfig().textvqa_root_dir)
    p.add_argument("--docvqa-root-dir", type=str, default=MarginalNetFullDatasetConfig().docvqa_root_dir)
    p.add_argument("--num-per-ds", type=int, default=MarginalNetFullDatasetConfig().num_per_ds)
    p.add_argument("--image-size", type=int, default=MarginalNetFullDatasetConfig().image_size)

    p.add_argument("--epochs", type=int, default=MarginalNetFullDatasetConfig().epochs)
    p.add_argument("--batch-size", type=int, default=MarginalNetFullDatasetConfig().batch_size)
    p.add_argument("--lr", type=float, default=MarginalNetFullDatasetConfig().lr)
    p.add_argument("--wd", type=float, default=MarginalNetFullDatasetConfig().wd)
    p.add_argument("--workers", type=int, default=MarginalNetFullDatasetConfig().workers)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--llava-repo-root", type=str, default=MarginalNetFullDatasetConfig().llava_repo_root)
    p.add_argument("--llava-model", type=str, default=MarginalNetFullDatasetConfig().llava_model)

    p.add_argument("--experiments-root", type=str, default=MarginalNetFullDatasetConfig().experiments_root)
    p.add_argument("--vis-every", type=int, default=MarginalNetFullDatasetConfig().vis_every)
    p.add_argument("--log-every", type=int, default=MarginalNetFullDatasetConfig().log_every)

    p.add_argument("--seed", type=int, default=MarginalNetFullDatasetConfig().seed)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = MarginalNetFullDatasetConfig(
        gqa_root_dir=args.gqa_root_dir,
        textvqa_root_dir=args.textvqa_root_dir,
        docvqa_root_dir=args.docvqa_root_dir,
        num_per_ds=args.num_per_ds,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        workers=args.workers,
        cpu=bool(args.cpu),
        llava_repo_root=args.llava_repo_root,
        llava_model=args.llava_model,
        experiments_root=args.experiments_root,
        vis_every=args.vis_every,
        log_every=args.log_every,
        seed=args.seed,
    )
    train_loop(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


