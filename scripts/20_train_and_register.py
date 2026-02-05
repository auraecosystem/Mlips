#!/usr/bin/env python3
"""
scripts/20_train_and_register.py

MNIST CNN demo:
- Train two small runs (run_a/run_b) with GitLab MLOps client experiment tracking
- Log params/metrics/artifacts
- Promote each run to GitLab Model Registry with a SemVer-compliant version string

Key fixes:
- No datetime.utcnow() (timezone-aware UTC everywhere)
- SemVer-safe model version strings (no ':' in build metadata)
- version is passed explicitly into promote_to_model_registry()
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

# GitLab MLOps Python Client
from gitlab_mlops import Client


# -------------------------
# helpers
# -------------------------

def utc_iso_z() -> str:
    """UTC timestamp in ISO 8601, Z-suffixed, timezone-aware."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def utc_compact() -> str:
    """
    Compact UTC timestamp safe for SemVer build metadata:
    only [0-9A-Za-z-] and no ':'.
    Example: 20260202T214119Z
    """
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def semver_version(run_name: str, base: str = "0.1.0") -> str:
    """
    Produce a SemVer-compliant version string that is unique per run.
    We use build metadata for uniqueness and traceability.
    Example: 0.1.0+20260202T214119Z.run_a
    """
    # run_name should be safe (letters/numbers/_). Replace underscores with hyphens to be safe.
    safe_run = "".join(ch if ch.isalnum() else "-" for ch in run_name).strip("-")
    return f"{base}+{utc_compact()}.{safe_run}"


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
    # Create run
    run = exp.create_run()
    if hasattr(run, "set_tag"):
        try:
            run.set_tag("run_name", run_name)
        except Exception:
            pass

    created_ts = utc_iso_z()
    version = semver_version(run_name=run_name, base="0.1.0")

    # Log params
    run.log_param("run_name", run_name)
    run.log_param("lr", lr)
    run.log_param("batch_size", batch_size)
    run.log_param("epochs", epochs)
    run.log_param("max_train", max_train)
    run.log_param("max_test", max_test)
    run.log_param("created_utc", created_ts)
    run.log_param("model_registry_version", version)

    # Link to dataset runs
    for k, v in (dataset_run_ids or {}).items():
        if v:
            run.log_param(f"dataset_{k}", v)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run.log_param("device", device)

    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

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

    # Save model
    out_dir = os.path.join("outputs", "models", run_name)
    ensure_dir(out_dir)
    model_path = os.path.join(out_dir, "model_state_dict.pt")
    torch.save(model.state_dict(), model_path)

    run.log_artifact(local_path=model_path, artifact_path="model")
    run.log_param("model_state_sha256", sha256_file(model_path))

    # Save run metadata artifact
    meta = {
        "run_name": run_name,
        "created_utc": created_ts,
        "hyperparams": {"lr": lr, "batch_size": batch_size, "epochs": epochs},
        "device": device,
        "model_registry_version": version,
        "source_run_id": getattr(run, "run_id", None),
    }
    meta_path = os.path.join(out_dir, "run_meta.json")
    write_json(meta_path, meta)
    run.log_artifact(local_path=meta_path, artifact_path="model")

    return run, model_path, meta_path, version


def promote_to_model_registry(client: Client, model_name: str, run, model_path: str, version: str):
    # Create or get model entry
    model = client.get_model(name=model_name)
    if not model:
        model = client.create_model(
            name=model_name,
            description="MNIST CNN demo model promoted from tracked runs.",
        )

    mv = model.create_version(
        description=f"Auto-promoted from run {getattr(run, 'run_id', 'unknown')}",
        version=version,
    )

    # Attach params; keep it minimal
    mv.log_param("source_run_id", getattr(run, "run_id", "unknown"))
    mv.log_param("artifact_path", "model/model_state_dict.pt")
    mv.log_param("model_state_sha256", sha256_file(model_path))
    mv.log_param("promoted_utc", utc_iso_z())

    # Add a small README to the model version
    mv.log_text(
        "This model version was produced by scripts/20_train_and_register.py.\n"
        "It is a small MNIST CNN intended for demonstrating GitLab experiment tracking + model registry.\n",
        "README.md",
    )

    # Artifacts remain logged to the run (MLflow tracking). Registry version stores a pointer to run_id.
    return mv


def main():
    cfg = load_config()
    client = Client(tracking_uri=cfg.tracking_uri, gitlab_token=cfg.token)

    # Training experiment
    exp = client.get_experiment(name=cfg.experiment_training) or client.create_experiment(cfg.experiment_training)

    # Pull dataset run pointers if available
    pointers_path = os.path.join("outputs", "datasets", "dataset_run_pointers.json")
    if os.path.exists(pointers_path):
        with open(pointers_path, "r", encoding="utf-8") as f:
            dataset_run_ids = json.load(f)
    else:
        dataset_run_ids = {"metadataset_run_id": None, "materialized_run_id": None}

    max_train = int(os.environ.get("MNIST_MAX_TRAIN", "2000"))
    max_test = int(os.environ.get("MNIST_MAX_TEST", "500"))

    # Two runs with different hyperparameters
    runspecs = [
        ("run_a", 1e-3, 64, 1),
        ("run_b", 5e-4, 128, 1),
    ]

    results = []
    for name, lr, bs, epochs in runspecs:
        run, model_path, meta_path, version = train_one_run(
            exp=exp,
            run_name=name,
            lr=lr,
            batch_size=bs,
            epochs=epochs,
            max_train=max_train,
            max_test=max_test,
            dataset_run_ids={
                "metadataset_run_id": dataset_run_ids.get("metadataset_run_id"),
                "materialized_run_id": dataset_run_ids.get("materialized_run_id"),
            },
        )
        mv = promote_to_model_registry(client, cfg.model_name, run, model_path, version)
        results.append(
            {
                "run_name": name,
                "run_id": getattr(run, "run_id", None),
                "model_version": getattr(mv, "version", None),
            }
        )

    out_summary = os.path.join("outputs", "training_summary.json")
    ensure_dir("outputs")
    write_json(out_summary, {"results": results})
    print("Training complete. Summary:", out_summary)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
