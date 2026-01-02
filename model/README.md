# Model Package Layout

This directory contains the training code for the MarginalNet full-dataset setup. Use the tree below as a map of how the modules connect.

```
model/
├── __init__.py
└── marginalnet_full_dataset/
    ├── __init__.py                -> Exports MarginalNetFullDatasetConfig + train_loop.
    ├── cli.py                     -> CLI entrypoint; builds config and calls trainer.train_loop.
    ├── config.py                  -> Dataclass for configuration defaults and env overrides.
    ├── trainer.py                 -> Orchestrates training loop; wires data, model, losses, logging.
    ├── data.py                    -> Dataset transforms/collation helpers used by trainer.
    ├── dataloader.py              -> Dataset definitions and sampling utilities (FullDataset).
    ├── model.py                   -> MarginalNet architecture + core tensor ops.
    ├── losses.py                  -> Loss functions used during training.
    ├── checkpoint_utils.py        -> Warp/CDF helpers, plotting utilities for checkpoints.
    ├── plots.py                   -> Training/validation curve plots.
    ├── experiment.py              -> Run directory creation and experiment bookkeeping.
    └── wandb_utils.py             -> Weights & Biases initialization and logging helpers.
```

## Environment activation

From the repository root, create the conda environment once and then activate it whenever you work on the model code:

```bash
conda env create -f attwarp.yaml
conda activate attwarp
```

If you already created the environment, just run the activation step.
