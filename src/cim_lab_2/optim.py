from __future__ import annotations

import torch


def build_optimizer(model, strategy: str, cfg):
    """Create explicit head/backbone parameter groups.

    STUDENT TASK 2:
    Inspect the groups and explain why the pretrained parameters use a smaller
    learning rate. Modify the groups only after the baseline runs work.
    """
    head_parameters = [p for p in model.fc.parameters() if p.requires_grad]
    backbone_parameters = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("fc.")
    ]

    groups = [{"params": head_parameters, "lr": float(cfg.head_lr), "name": "head"}]
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": float(cfg.backbone_lr),
                "name": "backbone",
            }
        )
    return torch.optim.AdamW(groups, weight_decay=float(cfg.weight_decay))
