"""Extension scaffold: nearest-prototype few-shot classification."""

from pathlib import Path
import hydra
from hydra.utils import to_absolute_path
import torch
from omegaconf import DictConfig

from .models import build_model, extract_features
from .run_io import load_run


def class_prototypes(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """STUDENT EXTENSION: return one mean feature vector per class."""
    raise NotImplementedError


def nearest_prototype(features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """STUDENT EXTENSION: classify by nearest prototype."""
    raise NotImplementedError


@hydra.main(version_base=None, config_path="../../conf", config_name="few_shot")
def main(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    saved_cfg, checkpoint = load_run(Path(to_absolute_path(cfg.run_dir)), device)
    model = build_model(saved_cfg.model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print("Loaded representation model. Complete the extension functions and support/query split.")
    print("Shots:", list(cfg.shots))


if __name__ == "__main__":
    main()
