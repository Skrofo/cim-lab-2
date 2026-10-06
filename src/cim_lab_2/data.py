from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def ensure_downloaded(root: str | Path) -> Path:
    """Use torchvision's downloader, then return the dataset directory."""
    OxfordIIITPet(root=str(root), split="trainval", download=True)
    dataset_dir = Path(root) / "oxford-iiit-pet"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Expected downloaded dataset at {dataset_dir}")
    return dataset_dir


def read_trainval_records(dataset_dir: Path) -> list[dict]:
    records = []
    split_file = dataset_dir / "annotations" / "trainval.txt"
    for line in split_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, class_id, species_id, breed_id = line.split()
        records.append(
            {
                "name": name,
                "breed": int(class_id) - 1,
                "species": int(species_id) - 1,
                "breed_id": int(breed_id) - 1,
            }
        )
    return records


def breed_names(records: list[dict], num_classes: int = 37) -> list[str]:
    names = [None] * num_classes
    for record in records:
        breed = record["breed"]
        name = record["name"].rsplit("_", 1)[0].replace("_", " ").title()
        if names[breed] is not None and names[breed] != name:
            raise ValueError(f"Conflicting names for breed index {breed}: {names[breed]} and {name}")
        names[breed] = name
    if any(name is None for name in names):
        raise ValueError("Could not determine a name for every breed")
    return names


def stratified_indices(records: list[dict], val_fraction: float, seed: int):
    """Create one deterministic train/validation split within every breed."""
    by_class = defaultdict(list)
    for index, record in enumerate(records):
        by_class[record["breed"]].append(index)

    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for class_indices in by_class.values():
        rng.shuffle(class_indices)
        n_val = max(1, round(len(class_indices) * val_fraction))
        val_indices.extend(class_indices[:n_val])
        train_indices.extend(class_indices[n_val:])
    return sorted(train_indices), sorted(val_indices)


def read_head_box(xml_path: Path) -> tuple[int, int, int, int]:
    root = ET.parse(xml_path).getroot()
    box = root.find("object/bndbox")
    if box is None:
        raise ValueError(f"No head bounding box in {xml_path}")
    xmin = int(float(box.findtext("xmin", "1")))
    ymin = int(float(box.findtext("ymin", "1")))
    xmax = int(float(box.findtext("xmax", "1")))
    ymax = int(float(box.findtext("ymax", "1")))
    return xmin, ymin, xmax, ymax


def expand_box(box, image_size, margin_fraction: float = 0.15):
    xmin, ymin, xmax, ymax = box
    width, height = image_size
    margin_x = int((xmax - xmin) * margin_fraction)
    margin_y = int((ymax - ymin) * margin_fraction)
    return (
        max(0, xmin - margin_x),
        max(0, ymin - margin_y),
        min(width, xmax + margin_x),
        min(height, ymax + margin_y),
    )


def build_transform(image_size: int, train: bool):
    if train:
        return v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop((image_size, image_size), scale=(0.75, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PetTransferDataset(Dataset):
    def __init__(self, dataset_dir, records, indices, view, transform):
        self.dataset_dir = Path(dataset_dir)
        self.records = records
        self.indices = list(indices)
        self.view = view
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        record_index = self.indices[item]
        record = self.records[record_index]
        image_path = self.dataset_dir / "images" / f"{record['name']}.jpg"
        image = Image.open(image_path).convert("RGB")

        if self.view == "head":
            xml_path = self.dataset_dir / "annotations" / "xmls" / f"{record['name']}.xml"
            box = expand_box(read_head_box(xml_path), image.size)
            image = image.crop(box)
        elif self.view != "full":
            raise ValueError(f"Unknown data view: {self.view}")

        image = self.transform(image)
        return image, record["breed"], record["species"], record_index


def limit_indices(indices, maximum, seed):
    if maximum is None or len(indices) <= maximum:
        return indices
    rng = random.Random(seed)
    return sorted(rng.sample(list(indices), int(maximum)))


def build_datasets(cfg, data_dir):
    dataset_dir = ensure_downloaded(data_dir)
    records = read_trainval_records(dataset_dir)
    train_indices, val_indices = stratified_indices(
        records, float(cfg.val_fraction), int(cfg.split_seed)
    )
    train_indices = limit_indices(train_indices, cfg.max_train_examples, cfg.split_seed)
    val_indices = limit_indices(val_indices, cfg.max_val_examples, cfg.split_seed + 1)

    train_dataset = PetTransferDataset(
        dataset_dir, records, train_indices, cfg.view, build_transform(cfg.image_size, True)
    )
    val_dataset = PetTransferDataset(
        dataset_dir, records, val_indices, cfg.view, build_transform(cfg.image_size, False)
    )
    class_to_species = torch.full((37,), -1, dtype=torch.long)
    for record in records:
        class_to_species[record["breed"]] = record["species"]
    if (class_to_species < 0).any():
        raise RuntimeError("Could not determine species for every breed")
    return train_dataset, val_dataset, class_to_species


def build_loaders(cfg, data_dir):
    train_dataset, val_dataset, class_to_species = build_datasets(cfg, data_dir)
    common = {
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "pin_memory": True,
        "persistent_workers": int(cfg.num_workers) > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader, class_to_species
