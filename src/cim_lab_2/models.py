from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

VALID_STRATEGIES = {"linear_probe", "partial", "full"}


def build_model(cfg) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if cfg.pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, int(cfg.num_classes))
    return model


def configure_trainable_parameters(model: nn.Module, strategy: str) -> None:
    """Select trainable parameters for the transfer-learning strategy.

    STUDENT TASK 1:
    Read this function before running the experiments. Add assertions or print
    statements that convince you the intended parameters are trainable.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy}; choose from {sorted(VALID_STRATEGIES)}")

    for parameter in model.parameters():
        parameter.requires_grad = False
        print("trainable parameter 1")


    for parameter in model.fc.parameters():
        parameter.requires_grad = True
        print("trainable parameter 2")

    if strategy in {"partial", "full"}:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    if strategy == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True


def set_transfer_mode(model: nn.Module, strategy: str, training: bool) -> None:
    """Set module modes while avoiding updates to frozen batch-normalisation statistics."""
    if not training:
        model.eval()
        return

    model.train()
    if strategy == "linear_probe":
        model.eval()
        model.fc.train()
    elif strategy == "partial":
        model.eval()
        model.layer4.train()
        model.fc.train()


def parameter_counts(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "parameters_total": total,
        "parameters_trainable": trainable,
        "trainable_fraction": trainable / total,
    }


def extract_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return the pooled ResNet representation before the classification head."""
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)
