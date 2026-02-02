# GitLab MLOps MNIST end-to-end demo (MVP)

This repository is an **approachable, script-first** demo of:

- GitLab **experiment tracking** (MLflow endpoint behind GitLab)
- GitLab **Model Registry**
- The **gitlab-mlops** Python client (`gitlab_mlops.Client`)
- A tiny MNIST workflow with:
  - **two dataset curation runs**
  - **two training runs**
  - promotion of each training run to a **Model Registry version**
  - a simple **inference** script that downloads a model artifact from MLflow

The goal is to be runnable by newcomers with minimal ceremony.

## Repo layout

```
.
├── scripts/
│   ├── 00_check_env.py
│   ├── 10_create_datasets.py
│   ├── 20_train_and_register.py
│   ├── 30_infer.py
│   └── util.py
├── docs/              # stub documentation pages
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+ recommended
- A GitLab project with:
  - Experiment Tracking enabled (GitLab-backed MLflow endpoint)
  - Model Registry enabled
- A GitLab access token with access to the project

## Configure authentication

Export environment variables (examples use gitlab.com; self-managed instances work the same way):

```bash
export MLFLOW_TRACKING_URI="https://gitlab.com/api/v4/projects/<project_id>/ml/mlflow"
export MLFLOW_TRACKING_TOKEN="your_access_token"
```

Optional:

```bash
export GITLAB_MLOPS_MODEL_NAME="mnist-cnn-demo"
export GITLAB_MLOPS_EXPERIMENT_DATASET="mnist_dataset_curation"
export GITLAB_MLOPS_EXPERIMENT_TRAINING="mnist_training"

# To keep the demo fast:
export MNIST_MAX_TRAIN=2000
export MNIST_MAX_TEST=500
```

Validate:

```bash
python scripts/00_check_env.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Dataset curation: 2 runs in one experiment

This creates two runs under the dataset experiment:

- Run 1: **metadataset only**
  - logs a manifest JSON artifact describing the MNIST subset
- Run 2: **metadataset + materialized dataset**
  - logs the same manifest
  - logs two tar.gz artifacts containing small train/test snapshots

```bash
python scripts/10_create_datasets.py
```

Outputs:
- `outputs/datasets/metadataset_manifest.json`
- `outputs/datasets/dataset_run_pointers.json` (links dataset run_ids for training)

## 2) Training + Model Registry promotion: 2 runs in one experiment

This creates two training runs with different hyperparameters and promotes each to the Model Registry.

```bash
python scripts/20_train_and_register.py
```

Output:
- `outputs/training_summary.json` (contains run_ids; you will use these for inference)

## 3) Inference: fetch model artifact by run_id

Pick a `run_id` from `outputs/training_summary.json` and run:

```bash
python scripts/30_infer.py --run-id <run_id> --n 16
```

## Notes / design choices

- This repo logs model weights as **MLflow artifacts** from the training runs.
- The promoted **Model Registry versions** include a pointer to the originating `run_id`,
  so downstream systems can fetch the exact bytes from MLflow.
- We keep packaging intentionally minimal (plain scripts + `requirements.txt`).

## Documentation stubs

See `docs/` for suggested documentation pages you would flesh out for real users:
- setup
- dataset curation
- training
- model registry
- troubleshooting
