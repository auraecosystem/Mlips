# Setup

## 1) Create a GitLab access token
You need a token that can access the project and its MLflow endpoint.

## 2) Export environment variables

Minimum required variables:

- `MLFLOW_TRACKING_URI`:
  `https://gitlab.com/api/v4/projects/<project_id>/ml/mlflow`
  (or your self-managed GitLab equivalent)

- `MLFLOW_TRACKING_TOKEN`:
  Your GitLab access token

Optional variables:

- `GITLAB_MLOPS_MODEL_NAME` (default: `mnist-cnn-demo`)
- `GITLAB_MLOPS_EXPERIMENT_DATASET` (default: `mnist_dataset_curation`)
- `GITLAB_MLOPS_EXPERIMENT_TRAINING` (default: `mnist_training`)
- `MNIST_MAX_TRAIN` / `MNIST_MAX_TEST` (for faster runs)

## 3) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
