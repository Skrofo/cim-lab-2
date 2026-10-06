import torch

from cim_lab_2.metrics import ClassificationAccumulator
from cim_lab_2.plots import save_confusion_matrix, save_per_breed_accuracy


def test_breed_and_species_accuracy_are_distinct():
    mapping = torch.tensor([0, 0, 1])
    accumulator = ClassificationAccumulator(3, mapping)
    logits = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    breeds = torch.tensor([0, 2])
    species = torch.tensor([0, 1])
    accumulator.update(logits, breeds, species)
    metrics = accumulator.compute()
    assert metrics["breed_accuracy"] == 0.5
    assert metrics["species_accuracy"] == 1.0


def test_breed_plots_are_saved_with_class_names(tmp_path):
    names = ["Abyssinian", "Bengal"]
    matrix_path = tmp_path / "confusion_matrix.png"
    accuracy_path = tmp_path / "per_breed_accuracy.png"

    save_confusion_matrix([[2, 0], [1, 1]], matrix_path, class_names=names)
    save_per_breed_accuracy([1.0, 0.5], names, accuracy_path)

    assert matrix_path.exists() and matrix_path.stat().st_size > 0
    assert accuracy_path.exists() and accuracy_path.stat().st_size > 0
