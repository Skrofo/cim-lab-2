from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(path, model, optimizer, epoch, metrics):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def save_resolved_config(path, cfg):
    OmegaConf.save(config=cfg, f=path, resolve=True)


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Expected resolved_config.yaml and best_checkpoint.pt in {run_dir}"
        )
    cfg = OmegaConf.load(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return cfg, checkpoint
