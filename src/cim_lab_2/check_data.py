from pathlib import Path
from types import SimpleNamespace
from .data import build_datasets


def main():
    cfg = SimpleNamespace(
        view="full", image_size=224, val_fraction=0.2, split_seed=42,
        max_train_examples=8, max_val_examples=8
    )
    train_dataset, val_dataset, mapping = build_datasets(cfg, Path("data"))
    image, breed, species, index = train_dataset[0]
    print("Train examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))
    print("Image shape:", tuple(image.shape))
    print("Breed/species/index:", breed, species, index)
    print("Breed-to-species mapping shape:", tuple(mapping.shape))


if __name__ == "__main__":
    main()
