#!/usr/bin/env python3
"""
scripts/20_train.py

MNIST CNN demo (training only):
- Creates runs under the training experiment
- Logs params/metrics/artifacts + run metadata
- DOES NOT create any Model Registry entries / versions

Promotion happens separately in scripts/30_promote.py.

Env controls:
- MNIST_MAX_TRAIN (default 2000)
- MNIST_MAX_TEST  (default 500)
- MNIST_EPOCHS    (default 10)
"""

import os
import json
from datetime import datetime, UTC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from util import load_config, ensure_dir, write_json, sha256_file
from gitlab_mlops import Client


# -------------------------
# helpers
# -------------------------

def utc_iso_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def utc_compact() -> str:
    # Safe characters for identifiers / filenames.
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def get_run_ref(run) -> str:
    """
    Different versions of gitlab_mlops client expose different identifiers.
    Prefer run_id, then external_id, then id.
    """
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
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "n": total,
    }


def train_one_run(
    exp,
    run_name: str,
    lr: float,
    batch_size: int,
    epochs: int,
    max_train: int,
    max_test: int,
    dataset_run_ids: dict,
):
    run = exp.create_run()

    # Optional tag for nicer browsing
    if hasattr(run, "set_tag"):
        try:
            run.set_tag("run_name", run_name)
        except Exception:
            pass

    created_ts = utc_iso_z()
    run_ref = get_run_ref(run)

    # Promotion status starts as "none"
    run.log_param("promotion_status", "none")
    run.log_param("created_utc", created_ts)
    run.log_param("run_name", run_name)
    run.log_param("lr", lr)
    run.log_param("batch_size", batch_size)
    run.log_param("epochs", epochs)
    run.log_param("max_train", max_train)
    run.log_param("max_test", max_test)
    run.log_param("run_ref", run_ref)

    # Link dataset provenance (if available)
    for k, v in (dataset_run_ids or {}).items():
        if v:
            run.log_param(f"dataset_{k}", v)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run.log_param("device", device)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

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
        # Some clients accept step, some don't
        try:
            run.log_metric("test_loss", float(metrics["loss"]), step=epoch)
            run.log_metric("test_accuracy", float(metrics["accuracy"]), step=epoch)
        except TypeError:
            run.log_metric("test_loss", float(metrics["loss"]))
            run.log_metric("test_accuracy", float(metrics["accuracy"]))

    # Save model + metadata artifacts under outputs/models/<run_name>_<timestamp>
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
        "source_run_ref": run_ref,
        "dataset_run_ids": dataset_run_ids or {},
        "model_state_sha256": model_sha,
        "artifact_path": "model/model_state_dict.pt",
    }
    meta_path = os.path.join(out_dir, "run_meta.json")
    write_json(meta_path, meta)
    run.log_artifact(local_path=meta_path, artifact_path="model")

    return {
        "run": run,
        "run_ref": run_ref,
        "run_name": run_name,
        "model_path": model_path,
        "meta_path": meta_path,
        "model_state_sha256": model_sha,
    }


def main():
    cfg = load_config()
    client = Client(tracking_uri=cfg.tracking_uri, gitlab_token=cfg.token)

    exp = client.get_experiment(name=cfg.experiment_training) or client.create_experiment(cfg.experiment_training)

    # Load dataset run pointers if they exist
    pointers_path = os.path.join("outputs", "datasets", "dataset_run_pointers.json")
    if os.path.exists(pointers_path):
        with open(pointers_path, "r", encoding="utf-8") as f:
            dataset_run_ids = json.load(f)
    else:
        dataset_run_ids = {"metadataset_run_id": None, "materialized_run_id": None}

    max_train = int(os.environ.get("MNIST_MAX_TRAIN", "2000"))
    max_test = int(os.environ.get("MNIST_MAX_TEST", "500"))
    epochs = int(os.environ.get("MNIST_EPOCHS", "10"))

    # Example: two runs with different hyperparameters
    runspecs = [
        ("run_a", 1e-3, 64, epochs),
        ("run_b", 5e-4, 128, epochs),
    ]

    results = []
    for name, lr, bs, ep in runspecs:
        r = train_one_run(
            exp=exp,
            run_name=name,
            lr=lr,
            batch_size=bs,
            epochs=ep,
            max_train=max_train,
            max_test=max_test,
            dataset_run_ids={
                "metadataset_run_id": dataset_run_ids.get("metadataset_run_id"),
                "materialized_run_id": dataset_run_ids.get("materialized_run_id"),
            },
        )
        results.append(
            {
                "run_name": r["run_name"],
                "run_ref": r["run_ref"],
                "model_state_sha256": r["model_state_sha256"],
            }
        )

    ensure_dir("outputs")
    out_summary = os.path.join("outputs", "training_summary.json")
    write_json(out_summary, {"results": results, "experiment": cfg.experiment_training})
    print("Training complete. Summary:", out_summary)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
