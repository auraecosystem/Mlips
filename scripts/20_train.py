#!/usr/bin/env python3
"""
scripts/20_train.py

MNIST CNN demo (training only):
- Creates runs under the training experiment
- Logs params/metrics/artifacts + run metadata
- DOES NOT create any Model Registry entries / versions

Dataset selection:
- REQUIRED: DATASET_RUN_ID=<run_id from dataset experiment>
- Optional: USE_MATERIALIZED_DATASET=true to train from dataset tarballs logged on that run

Env controls:
- DATASET_RUN_ID                (required)
- USE_MATERIALIZED_DATASET      (default false)
- MNIST_MAX_TRAIN               (default 2000)
- MNIST_MAX_TEST                (default 500)
- MNIST_EPOCHS                  (default 10)
"""

import os
import io
import json
import tarfile
from datetime import datetime, UTC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

import mlflow
from mlflow.tracking import MlflowClient

from util import load_config, ensure_dir, write_json, sha256_file, require_env
from gitlab_mlops import Client


# -------------------------
# helpers
# -------------------------

def utc_iso_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def utc_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def is_truthy(x) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def get_run_ref(run) -> str:
    # gitlab_mlops run objects differ by version
    return (
        getattr(run, "run_id", None)
        or getattr(run, "external_id", None)
        or getattr(run, "id", None)
        or "unknown"
    )


def safe_slug(s: str) -> str:
    s = (s or "").strip()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out).strip("-")
    return slug or "run"


# -------------------------
# optional: materialized dataset loading
# -------------------------

class MaterializedMNISTDataset(Dataset):
    """
    Loads .npy images + labels.json from the tarballs produced by create_dataset.py.
    """
    def __init__(self, tar_path: str):
        self.samples = []  # list of (np.ndarray, int)
        with tarfile.open(tar_path, "r:gz") as tf:
            labels_member = None
            for m in tf.getmembers():
                if m.name.endswith("/labels.json"):
                    labels_member = m
                    break
            if labels_member is None:
                raise RuntimeError(f"No labels.json found in {tar_path}")

            labels = json.loads(tf.extractfile(labels_member).read().decode("utf-8"))

            for name, y in sorted(labels.items()):
                if not name.endswith(".npy"):
                    continue
                f = tf.extractfile(name)
                if f is None:
                    continue
                b = f.read()
                arr = np.load(io.BytesIO(b), allow_pickle=False)
                self.samples.append((arr, int(y)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr, y = self.samples[idx]
        x = torch.from_numpy(arr)

        # arr may be HxW (28x28) OR 1xHxW (1x28x28) depending on how it was saved
        if x.ndim == 2:
            x = x.unsqueeze(0)  # -> 1xHxW
        elif x.ndim == 3 and x.shape[0] != 1:
            # If someone saved HxWxC, convert to CxHxW
            x = x.permute(2, 0, 1)

        x = x.float() / 255.0
        return x, torch.tensor(int(y), dtype=torch.long)



def download_materialized_tarballs(mlc: MlflowClient, dataset_run_id: str, out_dir: str):
    """
    Downloads train/test tarballs from the dataset run artifacts:
      dataset/materialized/mnist_small_train.tar.gz
      dataset/materialized/mnist_small_test.tar.gz
    """
    ensure_dir(out_dir)
    train_tar = mlc.download_artifacts(dataset_run_id, "dataset/materialized/mnist_small_train.tar.gz", out_dir)
    test_tar  = mlc.download_artifacts(dataset_run_id, "dataset/materialized/mnist_small_test.tar.gz", out_dir)
    return train_tar, test_tar


# -------------------------
# model
# -------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1), "n": total}


def train_one_run(exp, cfg, mlc: MlflowClient, dataset_run_id: str,
                  run_name: str, lr: float, batch_size: int, epochs: int,
                  max_train: int, max_test: int):
    run = exp.create_run()

    # Optional tag for nicer browsing
    if hasattr(run, "set_tag"):
        try:
            run.set_tag("run_name", run_name)
        except Exception:
            pass

    created_ts = utc_iso_z()
    run_ref = get_run_ref(run)

    # ---- dataset provenance (required) ----
    ds = mlc.get_run(dataset_run_id)
    dataset_kind = ds.data.params.get("dataset_kind")

    run.log_param("dataset_experiment", cfg.experiment_dataset)
    run.log_param("dataset_run_id", dataset_run_id)
    if dataset_kind:
        run.log_param("dataset_kind", dataset_kind)

    # copy a few useful dataset params/hashes if present
    for k in ("manifest_sha256", "train_tar_sha256", "test_tar_sha256", "max_train", "max_test"):
        v = ds.data.params.get(k)
        if v is not None:
            run.log_param(f"dataset_{k}", v)

    # ---- training params ----
    run.log_param("promotion_status", "none")
    run.log_param("created_utc", created_ts)
    run.log_param("run_ref", run_ref)
    run.log_param("run_name", run_name)
    run.log_param("lr", lr)
    run.log_param("batch_size", batch_size)
    run.log_param("epochs", epochs)
    run.log_param("max_train", max_train)
    run.log_param("max_test", max_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run.log_param("device", device)

    # ---- choose data source ----
    use_materialized = is_truthy(os.environ.get("USE_MATERIALIZED_DATASET", "false"))
    if use_materialized:
        if dataset_kind != "metadataset_plus_materialized":
            raise RuntimeError(
                "USE_MATERIALIZED_DATASET=true requires dataset_kind=metadataset_plus_materialized "
                f"but selected dataset_kind={dataset_kind!r}"
            )
        dl_dir = os.path.join("outputs", "datasets_downloaded", dataset_run_id)
        train_tar, test_tar = download_materialized_tarballs(mlc, dataset_run_id, dl_dir)
        train_ds = MaterializedMNISTDataset(train_tar)
        test_ds = MaterializedMNISTDataset(test_tar)
        run.log_param("dataset_source", "materialized_tarballs")
    else:
        tfm = transforms.Compose([transforms.ToTensor()])
        train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
        run.log_param("dataset_source", "torchvision_mnist_download")

    # Subset for speed
    train_subset = Subset(train_ds, list(range(min(len(train_ds), max_train))))
    test_subset = Subset(test_ds, list(range(min(len(test_ds), max_test))))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    # Initial eval
    metrics0 = evaluate(model, test_loader, device)
    run.log_metric("test_loss", float(metrics0["loss"]))
    run.log_metric("test_accuracy", float(metrics0["accuracy"]))

    # Train
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"{run_name} epoch {epoch}/{epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        metrics = evaluate(model, test_loader, device)
        try:
            run.log_metric("test_loss", float(metrics["loss"]), step=epoch)
            run.log_metric("test_accuracy", float(metrics["accuracy"]), step=epoch)
        except TypeError:
            run.log_metric("test_loss", float(metrics["loss"]))
            run.log_metric("test_accuracy", float(metrics["accuracy"]))

    # Save model + metadata artifacts
    out_dir = os.path.join("outputs", "models", f"{safe_slug(run_name)}_{utc_compact()}")
    ensure_dir(out_dir)

    model_path = os.path.join(out_dir, "model_state_dict.pt")
    torch.save(model.state_dict(), model_path)
    model_sha = sha256_file(model_path)

    run.log_artifact(local_path=model_path, artifact_path="model")
    run.log_param("model_state_sha256", model_sha)
    run.log_param("artifact_path_model_state", "model/model_state_dict.pt")

    meta = {
        "run_name": run_name,
        "created_utc": created_ts,
        "hyperparams": {"lr": lr, "batch_size": batch_size, "epochs": epochs},
        "device": device,
        "dataset": {
            "experiment": cfg.experiment_dataset,
            "run_id": dataset_run_id,
            "kind": dataset_kind,
            "source": "materialized_tarballs" if use_materialized else "torchvision_mnist_download",
        },
        "model_state_sha256": model_sha,
        "artifact_path": "model/model_state_dict.pt",
        "source_run_ref": run_ref,
    }
    meta_path = os.path.join(out_dir, "run_meta.json")
    write_json(meta_path, meta)
    run.log_artifact(local_path=meta_path, artifact_path="model")

    return {"run_name": run_name, "run_ref": run_ref, "model_state_sha256": model_sha}


def main():
    cfg = load_config()
    dataset_run_id = require_env("DATASET_RUN_ID")

    client = Client(tracking_uri=cfg.tracking_uri, gitlab_token=cfg.token)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlc = MlflowClient()

    # Validate dataset run exists early
    _ = mlc.get_run(dataset_run_id)

    exp = client.get_experiment(name=cfg.experiment_training) or client.create_experiment(cfg.experiment_training)

    max_train = int(os.environ.get("MNIST_MAX_TRAIN", "2000"))
    max_test = int(os.environ.get("MNIST_MAX_TEST", "500"))
    epochs = int(os.environ.get("MNIST_EPOCHS", "10"))

    runspecs = [
        ("run_a", 1e-3, 64, epochs),
        ("run_b", 5e-4, 128, epochs),
    ]

    results = []
    for name, lr, bs, ep in runspecs:
        results.append(
            train_one_run(
                exp=exp,
                cfg=cfg,
                mlc=mlc,
                dataset_run_id=dataset_run_id,
                run_name=name,
                lr=lr,
                batch_size=bs,
                epochs=ep,
                max_train=max_train,
                max_test=max_test,
            )
        )

    ensure_dir("outputs")
    out_summary = os.path.join("outputs", "training_summary.json")
    write_json(out_summary, {"results": results, "experiment": cfg.experiment_training})
    print("Training complete. Summary:", out_summary)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
