from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class ClassificationAccumulator:
    def __init__(self, num_classes: int, class_to_species: torch.Tensor):
        self.num_classes = num_classes
        self.class_to_species = class_to_species.cpu()
        self.targets = []
        self.predictions = []
        self.species_targets = []
        self.species_predictions = []

    def update(self, logits, breed_targets, species_targets):
        breed_predictions = logits.argmax(dim=1).detach().cpu()
        breed_targets = breed_targets.detach().cpu()
        species_targets = species_targets.detach().cpu()
        species_predictions = self.class_to_species[breed_predictions]
        self.targets.extend(breed_targets.tolist())
        self.predictions.extend(breed_predictions.tolist())
        self.species_targets.extend(species_targets.tolist())
        self.species_predictions.extend(species_predictions.tolist())

    def compute(self):
        targets = np.asarray(self.targets)
        predictions = np.asarray(self.predictions)
        species_targets = np.asarray(self.species_targets)
        species_predictions = np.asarray(self.species_predictions)
        matrix = confusion_matrix(targets, predictions, labels=np.arange(self.num_classes))
        totals = matrix.sum(axis=1)
        per_class = np.divide(
            np.diag(matrix),
            totals,
            out=np.full(self.num_classes, np.nan, dtype=float),
            where=totals > 0,
        )

        # STUDENT TASK 3:
        # Add at least one additional diagnostic that helps reveal uneven
        # performance across breeds. Keep the existing keys so training runs.
        return {
            "breed_accuracy": float((targets == predictions).mean()),
            "species_accuracy": float((species_targets == species_predictions).mean()),
            "macro_breed_accuracy": float(np.nanmean(per_class)),
            "per_class_accuracy": per_class.tolist(),
            "confusion_matrix": matrix.tolist(),
        }
