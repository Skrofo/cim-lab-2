from __future__ import annotations

from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .data import build_loaders
from .models import build_model, configure_trainable_parameters
from .run_io import load_run
from .train import run_epoch


@hydra.main(version_base=None, config_path="../../conf", config_name="evaluate")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = Path(to_absolute_path(cfg.run_dir))
    saved_cfg, checkpoint = load_run(run_dir, device)
    model = build_model(saved_cfg.model)
    configure_trainable_parameters(model, saved_cfg.strategy.name)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    merged = OmegaConf.merge(
        saved_cfg.data,
        {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers},
    )
    _, val_loader, class_to_species = build_loaders(merged, saved_cfg.data_dir)
    metrics = run_epoch(
        model, val_loader, nn.CrossEntropyLoss(), device, class_to_species,
        saved_cfg.strategy.name, optimizer=None, amp=saved_cfg.training.amp
    )
    print(metrics)


if __name__ == "__main__":
    main()
