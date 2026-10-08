# CIM Lab 2: Transfer Learning and Dataset Dependence

This lab uses a pretrained ResNet-18 to classify the 37 breeds in the Oxford-IIIT Pet Dataset. You will compare linear probing, partial fine-tuning and full fine-tuning, then test whether conclusions change when the model sees the full image or only the annotated head region.

## Learning objectives

By the end of the lab, you should be able to:

- use an ImageNet-pretrained model for a new classification task;
- control which model parameters are trainable;
- use different learning rates for a pretrained backbone and a new classification head;
- compare transfer-learning strategies under a fixed compute budget;
- evaluate breed accuracy, species accuracy and per-class performance;
- investigate how image content and dataset construction affect conclusions;
- preserve configurations, curves and checkpoints using Hydra and Weights & Biases.

## Setup and guidance

- [Lab 0 setup and WSL guidance](https://github.com/ivorsimpson/cim-lab-0/blob/main/README.md)
- [Lab 0 GitHub support](https://github.com/ivorsimpson/cim-lab-0/blob/main/docs/github-support.md)
- [Lab 1 repository and workflow](https://github.com/ivorsimpson/cim-lab-1/)


The dataset is downloaded automatically. ResNet-18 ImageNet weights are downloaded on first use.

Make sure you clone this repository to start with and move into the correct directory

## Running Python
Synchronise the environment:

```bash
uv sync
```

All commands must run through `uv`:

```bash
uv run python -m cim_lab_2.train
```

or after activating the project environment:

```bash
source .venv/bin/activate
python -m cim_lab_2.train
```

Do not install packages globally.

## Before training

Log in to Weights & Biases:

```bash
uv run wandb login
```

Important runs should be uploaded before leaving the lab because local WSL storage may not persist

Run the fast checks:

```bash
uv run pytest
uv run python -m cim_lab_2.check_data
```

## Compute budget

The default configuration uses:

- ResNet-18 with ImageNet weights;
- 224 x 224 inputs;
- five epochs;
- automatic mixed precision on CUDA;
- one GPU;
- a stratified 80/20 train/validation split from the dataset's `trainval` records.

The split is deterministic and shared across the full-image and head-crop runs. The official Oxford-IIIT Pet test split is not used, so report these results as validation performance rather than official test performance.

Use debug mode while editing:

```bash
uv run python -m cim_lab_2.train experiment=debug run.name=debug
```

Do not run large sweeps as it will take a long time. The core comparison is three runs.

## Core workflow

### 1. Inspect the transfer-learning code

Start by reading:

1. `src/cim_lab_2/models.py`
2. `src/cim_lab_2/optim.py`

Search for `STUDENT TASK` to find the exact places where you should write or modify code.

Take a look at `src/cim_lab_2/data.py`, `src/cim_lab_2/metrics.py` and `src/cim_lab_2/train.py`. You don't need to understand every detail, but make sure you understand their roles in the training and evaluation pipeline.

### 2. Linear probing

Train only the new classification head:

```bash
uv run python -m cim_lab_2.train \
  strategy=linear_probe \
  run.name=full_linear_probe
```

Inspect `configure_trainable_parameters()` in `models.py`. Record the number and proportion of trainable parameters.

### 3. Partial fine-tuning

Train the classification head and the final ResNet block:

```bash
uv run python -m cim_lab_2.train \
  strategy=partial \
  run.name=full_partial
```

Inspect the parameter groups in `optim.py`. Why is the backbone learning rate smaller than the head learning rate?

### 4. Full fine-tuning

Train the complete model:

```bash
uv run python -m cim_lab_2.train \
  strategy=full \
  run.name=full_finetune
```

After completing all three runs, compare:

- the number and proportion of trainable parameters;
- training time;
- validation accuracy;
- the generalisation gap (difference between training and validation accuracy).

Which performs best? Does training more parameters always improve validation accuracy?

### 5. Evaluate the task at two levels

The network predicts one of 37 breeds. The code also converts each predicted breed into a cat/dog prediction.

The existing metrics code already reports:

- breed accuracy;
- cat/dog species accuracy;
- per-breed accuracy;
- macro-average breed accuracy;
- a confusion matrix.

Inspect the marked section in `metrics.py` and add one further diagnostic that helps explain uneven performance across breeds. 

Each run also saves a breed-named per-class accuracy plot and JSON file, plus a breed-labelled confusion matrix.

Ask: can a model perform well on cat-versus-dog while remaining poor at fine-grained breed recognition?

### 6. Investigate dataset dependence

Full images contain information about the animal's body and background, which may influence predictions. To test this, repeat one transfer strategy using annotated head crops:

```bash
uv run python -m cim_lab_2.train \
  strategy=partial \
  data.view=head \
  run.name=head_partial
```

Compare this with `full_partial`.

Consider:

- Does the full image provide useful body-shape information?
- Could background, framing or scale provide shortcuts?
- Does a head crop remove useful information as well as nuisance information?
- Which breeds benefit or deteriorate most?

### 7. Compare runs in W&B

Each run logs:

- the complete resolved configuration;
- train and validation loss curves;
- breed and species accuracy;
- learning rates,trainable parameter count and runtime;
- the best checkpoint, configuration, per-breed results and plots as a model artifact.

Use the W&B comparison view to compare your core runs. Before leaving, verify that the important runs and artifacts have synchronised successfully.

## Where results are stored

Hydra creates a distinct local directory for every run:

```text
outputs/runs/<run-name>/<timestamp>/
├── .hydra/
├── best_checkpoint.pt
├── resolved_config.yaml
├── summary.json
├── per_breed_accuracy.json
├── per_breed_accuracy.png
├── confusion_matrix.png
└── wandb_run.json  # present only when W&B logging is enabled
```

These local files are useful during the lab but should not be treated as permanent storage.

Reload and evaluate a run directory with:

```bash
uv run python -m cim_lab_2.evaluate \
  run_dir=outputs/runs/full_partial/<timestamp>
```

## Minimum completion

- [ ] Run the automated tests and dataset check.
- [ ] Complete linear probing, partial fine-tuning and full fine-tuning.
- [ ] Compare trainable parameter counts and runtime.
- [ ] Report breed and species accuracy.
- [ ] Examine per-breed performance and a confusion matrix.
- [ ] Compare full-image and head-crop inputs for one strategy.
- [ ] Confirm that important runs and checkpoints are stored in W&B.
- [ ] Commit code and configuration to GitHub.

## Extensions

### Random initialisation

```bash
uv run python -m cim_lab_2.train \
  strategy=full \
  model.pretrained=false \
  run.name=random_initialisation
```

### Few-shot classification

Optional extension for you to implement in your own time; it is not part of the core lab. `few_shot.py` is a scaffold: implement feature extraction and a support/query split, then compare nearest-prototype classification with 1, 5 and 10 labelled examples per held-out breed. This follows the idea of [Prototypical Networks for Few-shot Learning (Snell, Swersky and Zemel, 2017)](https://arxiv.org/abs/1703.05175).

### Foreground-only input

The dataset provides trimap segmentation annotations. Build a foreground-only view and compare it with full images and head crops.

### Waterbirds

Read about the Waterbirds benchmark as a stronger example of background correlation and group shift. Treat this as a conceptual extension; dataset plumbing is intentionally not included in the core lab.

## Further Resources and Documentation

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [Torchvision ResNet-18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
- [Hydra output-directory guidance](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/)
- [Weights & Biases quickstart](https://docs.wandb.ai/models/quickstart)
- [Weights & Biases artifacts](https://docs.wandb.ai/models/artifacts)
