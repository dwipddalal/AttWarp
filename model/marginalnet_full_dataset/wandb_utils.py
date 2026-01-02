from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict


class DummyWandbRun:
    def log(self, *a, **k):
        pass

    def watch(self, *a, **k):
        pass

    def define_metric(self, *a, **k):
        pass

    def log_artifact(self, *a, **k):
        pass

    def finish(self):
        pass

    @property
    def summary(self):
        return {}


def init_wandb(
    *,
    run_dir: Path,
    project: str,
    entity: str | None,
    cfg: Dict[str, Any],
) -> Any:
    """
    Initialize wandb with fallback to offline/disabled (for cluster permission issues).
    Returns a wandb run-like object.
    """
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

    mode = os.getenv("WANDB_MODE", "online")
    print(f"[wandb] init: project={project} entity={entity or '(default)'} mode={mode}")

    try:
        import wandb

        run = wandb.init(
            project=project,
            entity=entity or None,
            name=run_dir.name,
            dir=str(run_dir),
            config=cfg,
        )
        run.define_metric("global_step")
        run.define_metric("epoch")
        run.define_metric("loss/*", step_metric="global_step")
        run.define_metric("grad_norm", step_metric="global_step")
        return run

    except Exception as e1:
        msg = str(e1)
        if "PERMISSION_ERROR" in msg or "403" in msg or "permission denied" in msg.lower():
            print("[wandb] permission error; forcing OFFLINE mode for this run.")
            os.environ["WANDB_MODE"] = "offline"
            try:
                import wandb

                run = wandb.init(
                    project=project,
                    name=run_dir.name,
                    dir=str(run_dir),
                    config=cfg,
                    settings=wandb.Settings(mode="offline"),
                )
                run.define_metric("global_step")
                run.define_metric("epoch")
                run.define_metric("loss/*", step_metric="global_step")
                run.define_metric("grad_norm", step_metric="global_step")
                return run
            except Exception as e2:
                print("[wandb] offline init still failed; disabling W&B. Error:\n", e2)
                os.environ["WANDB_DISABLED"] = "true"
                return DummyWandbRun()

        print("[wandb] unexpected init failure; disabling W&B. Error:\n", e1)
        os.environ["WANDB_DISABLED"] = "true"
        return DummyWandbRun()


