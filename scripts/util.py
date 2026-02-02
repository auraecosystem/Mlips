import os
import json
import hashlib
from dataclasses import dataclass
from typing import Optional

def require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

@dataclass
class DemoConfig:
    tracking_uri: str
    token: str
    experiment_dataset: str
    experiment_training: str
    model_name: str

def load_config() -> DemoConfig:
    # These are consistent with the GitLab MLOps python client README.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    token = os.environ.get("MLFLOW_TRACKING_TOKEN") or os.environ.get("GITLAB_TOKEN")

    if not tracking_uri:
        raise RuntimeError("Missing MLFLOW_TRACKING_URI")
    if not token:
        raise RuntimeError("Missing MLFLOW_TRACKING_TOKEN (or GITLAB_TOKEN)")

    return DemoConfig(
        tracking_uri=tracking_uri,
        token=token,
        experiment_dataset=os.environ.get("GITLAB_MLOPS_EXPERIMENT_DATASET", "mnist_dataset_curation"),
        experiment_training=os.environ.get("GITLAB_MLOPS_EXPERIMENT_TRAINING", "mnist_training"),
        model_name=os.environ.get("GITLAB_MLOPS_MODEL_NAME", "mnist-cnn-demo"),
    )
