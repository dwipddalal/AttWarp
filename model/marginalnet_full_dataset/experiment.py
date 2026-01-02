from __future__ import annotations

import shutil
import time
from pathlib import Path


def _next_experiment_dir_name(root: Path) -> str:
    max_idx = 0
    if root.exists():
        for d in root.iterdir():
            if d.is_dir() and d.name.startswith("Experiment_"):
                try:
                    idx = int(d.name.split("_")[-1])
                except Exception:
                    continue
                max_idx = max(max_idx, idx)
    return f"Experiment_{max_idx + 1}"


def create_experiment_run_dir(experiments_root: str, project_root: str) -> Path:
    base_root = Path(experiments_root)
    exp_name = _next_experiment_dir_name(base_root)
    exp_dir = base_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "debug").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "code_snapshot").mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "comments.txt", "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

    pr = Path(project_root)
    for py_file in pr.glob("*.py"):
        shutil.copy2(str(py_file), exp_dir / "code_snapshot" / py_file.name)

    return exp_dir


