from __future__ import annotations

import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .data import breed_names, build_loaders
from .metrics import ClassificationAccumulator
from .models import (
    build_model,
    configure_trainable_parameters,
    parameter_counts,
    set_transfer_mode,
)
from .optim import build_optimizer
from .plots import save_confusion_matrix, save_per_breed_accuracy
from .run_io import save_checkpoint, save_resolved_config, write_json


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, class_to_species, strategy, optimizer=None, amp=True):
    training = optimizer is not None
    set_transfer_mode(model, strategy, training)
    accumulator = ClassificationAccumulator(37, class_to_species)
    total_loss = 0.0
    total_examples = 0
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for images, breeds, species, _ in loader:
            images = images.to(device, non_blocking=True)
            breeds = breeds.to(device, non_blocking=True)
            species = species.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = loss_fn(logits, breeds)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * breeds.size(0)
            total_examples += breeds.size(0)
            accumulator.update(logits, breeds, species)

    metrics = accumulator.compute()
    metrics["loss"] = total_loss / total_examples
    return metrics


def wandb_config(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(int(cfg.seed))
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(run_dir / "resolved_config.yaml", cfg)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    train_cfg = OmegaConf.merge(cfg.data, cfg.training)
    train_loader, val_loader, class_to_species = build_loaders(train_cfg, cfg.data_dir)
    class_names = breed_names(train_loader.dataset.records)

    model = build_model(cfg.model)
    configure_trainable_parameters(model, cfg.strategy.name)
    model.to(device)
    counts = parameter_counts(model)
    optimizer = build_optimizer(model, cfg.strategy.name, cfg.training)
    loss_fn = nn.CrossEntropyLoss()

    wb_run = None
    if cfg.logging.use_wandb:
        import wandb
        display_name = None if cfg.run.name == "auto" else cfg.run.name
        wb_run = wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=display_name,
            config=wandb_config(cfg),
            dir=str(run_dir),
            job_type="training",
            tags=[cfg.strategy.name, cfg.data.view],
        )
        write_json(
            run_dir / "wandb_run.json",
            {"id": wb_run.id, "name": wb_run.name, "project": wb_run.project, "entity": wb_run.entity},
        )

    best_accuracy = -1.0
    best_metrics = None
    start = time.perf_counter()
    for epoch in range(1, int(cfg.training.epochs) + 1):
        train_metrics = run_epoch(
            model, train_loader, loss_fn, device, class_to_species,
            cfg.strategy.name, optimizer, cfg.training.amp
        )
        val_metrics = run_epoch(
            model, val_loader, loss_fn, device, class_to_species,
            cfg.strategy.name, None, cfg.training.amp
        )
        log = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/breed_accuracy": train_metrics["breed_accuracy"],
            "train/species_accuracy": train_metrics["species_accuracy"],
            "val/loss": val_metrics["loss"],
            "val/breed_accuracy": val_metrics["breed_accuracy"],
            "val/species_accuracy": val_metrics["species_accuracy"],
            "val/macro_breed_accuracy": val_metrics["macro_breed_accuracy"],
        }
        for index, group in enumerate(optimizer.param_groups):
            log[f"lr/{group.get('name', index)}"] = group["lr"]
        print(log)
        if wb_run is not None:
            wb_run.log(log)

        if val_metrics["breed_accuracy"] > best_accuracy:
            best_accuracy = val_metrics["breed_accuracy"]
            best_metrics = val_metrics
            save_checkpoint(
                run_dir / "best_checkpoint.pt", model, optimizer, epoch, val_metrics
            )

    runtime = time.perf_counter() - start
    summary = {
        **counts,
        "runtime_seconds": runtime,
        "best_val_breed_accuracy": best_accuracy,
        "best_val_species_accuracy": best_metrics["species_accuracy"],
        "best_val_macro_breed_accuracy": best_metrics["macro_breed_accuracy"],
        "strategy": cfg.strategy.name,
        "data_view": cfg.data.view,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "per_breed_accuracy.json",
        dict(zip(class_names, best_metrics["per_class_accuracy"])),
    )
    save_confusion_matrix(
        best_metrics["confusion_matrix"],
        run_dir / "confusion_matrix.png",
        class_names=class_names,
    )
    save_per_breed_accuracy(
        best_metrics["per_class_accuracy"],
        class_names,
        run_dir / "per_breed_accuracy.png",
    )

    if wb_run is not None:
        wb_run.summary.update(summary)
        if cfg.logging.log_model_artifact:
            import wandb
            artifact = wandb.Artifact(
                name=f"pets-{cfg.strategy.name}-{cfg.data.view}-{wb_run.id}",
                type="model",
                metadata=summary,
            )
            for filename in [
                "best_checkpoint.pt", "resolved_config.yaml", "summary.json",
                "confusion_matrix.png", "per_breed_accuracy.json", "per_breed_accuracy.png",
            ]:
                artifact.add_file(str(run_dir / filename), name=filename)
            wb_run.log_artifact(artifact)
        wb_run.finish()

    print(f"Run directory: {run_dir}")
    print(summary)


if __name__ == "__main__":
    main()
