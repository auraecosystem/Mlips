import os
import io
import json
import tarfile
from datetime import datetime, UTC
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms

from util import load_config, ensure_dir, write_json, sha256_bytes, sha256_file

# GitLab MLOps Python Client
from gitlab_mlops import Client


def build_manifest(ds, split: str, max_items: int) -> List[Dict]:
    items = []
    n = min(len(ds), max_items)
    for i in range(n):
        img, label = ds[i]
        # torchvision transforms may return torch tensors; we keep bytes hashing simple:
        # convert to raw bytes using numpy.
        arr = np.array(img, dtype=np.uint8)
        b = arr.tobytes()
        items.append({
            "sample_id": f"{split}:{i}",
            "split": split,
            "label": int(label),
            "shape": list(arr.shape),
            "sha256": sha256_bytes(b),
        })
    return items


def materialize_dataset_tar(ds, split: str, max_items: int, out_path: str) -> None:
    """Write a small tar.gz containing .npy arrays and a labels.json."""
    n = min(len(ds), max_items)
    labels = {}
    with tarfile.open(out_path, "w:gz") as tf:
        for i in tqdm(range(n), desc=f"materialize-{split}"):
            img, label = ds[i]
            arr = np.array(img, dtype=np.uint8)
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            buf.seek(0)
            name = f"{split}/img_{i:06d}.npy"
            info = tarfile.TarInfo(name=name)
            data = buf.getvalue()
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            labels[name] = int(label)

        # labels json
        meta = io.BytesIO(json.dumps(labels, indent=2, sort_keys=True).encode("utf-8"))
        info = tarfile.TarInfo(name=f"{split}/labels.json")
        info.size = len(meta.getvalue())
        tf.addfile(info, meta)


def main():
    cfg = load_config()
    client = Client(tracking_uri=cfg.tracking_uri, gitlab_token=cfg.token)

    out_dir = os.path.join("outputs", "datasets")
    ensure_dir(out_dir)

    max_train = int(os.environ.get("MNIST_MAX_TRAIN", "2000"))
    max_test  = int(os.environ.get("MNIST_MAX_TEST", "500"))

    tfm = transforms.ToTensor()  # keeps it simple; hashing uses numpy conversion above
    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

    # -------------------------
    # Experiment: dataset curation
    # -------------------------
    exp = client.get_experiment(name=cfg.experiment_dataset) or client.create_experiment(cfg.experiment_dataset)

    # Run 1: metadataset only
    run1 = exp.create_run()
    run1.log_param("dataset_kind", "metadataset_only")
    run1.log_param("max_train", max_train)
    run1.log_param("max_test", max_test)
    run1.log_param("created_utc", datetime.now(UTC).isoformat().replace("+00:00", "Z"))

    manifest_train = build_manifest(train_ds, "train", max_train)
    manifest_test = build_manifest(test_ds, "test", max_test)
    manifest = {
        "name": "mnist_small",
        "version": "v1",
        "notes": "Metadataset manifest for a small MNIST subset.",
        "splits": {
            "train": {"count": len(manifest_train)},
            "test": {"count": len(manifest_test)},
        },
        "items": manifest_train + manifest_test,
    }

    manifest_path = os.path.join(out_dir, "metadataset_manifest.json")
    write_json(manifest_path, manifest)

    run1.log_artifact(local_path=manifest_path, artifact_path="dataset")
    run1.log_param("manifest_sha256", sha256_file(manifest_path))
    run1.log_metric("n_train", float(len(manifest_train)))
    run1.log_metric("n_test", float(len(manifest_test)))

    # Run 2: metadataset + materialized dataset artifact
    run2 = exp.create_run()
    run2.log_param("dataset_kind", "metadataset_plus_materialized")
    run2.log_param("max_train", max_train)
    run2.log_param("max_test", max_test)
    run2.log_param("created_utc", datetime.now(UTC).isoformat().replace("+00:00", "Z"))

    # reuse manifest but log separately so it is tied to run2
    run2.log_artifact(local_path=manifest_path, artifact_path="dataset")
    run2.log_param("manifest_sha256", sha256_file(manifest_path))

    tar_path = os.path.join(out_dir, "mnist_small_materialized.tar.gz")
    materialize_dataset_tar(train_ds, "train", max_train, tar_path)
    materialize_dataset_tar(test_ds, "test", max_test, tar_path)  # NOTE: appends? no; overwrite
    # To keep it simple and safe, write two tarballs instead.
    # We'll do that by rewriting:
    tar_train = os.path.join(out_dir, "mnist_small_train.tar.gz")
    tar_test  = os.path.join(out_dir, "mnist_small_test.tar.gz")
    materialize_dataset_tar(train_ds, "train", max_train, tar_train)
    materialize_dataset_tar(test_ds, "test", max_test, tar_test)

    run2.log_artifact(local_path=tar_train, artifact_path="dataset/materialized")
    run2.log_artifact(local_path=tar_test, artifact_path="dataset/materialized")
    run2.log_param("train_tar_sha256", sha256_file(tar_train))
    run2.log_param("test_tar_sha256", sha256_file(tar_test))

    run2.log_metric("n_train", float(max_train))
    run2.log_metric("n_test", float(max_test))

    # Emit a small pointer file so training can be linked to dataset runs easily.
    pointers = {
        "dataset_experiment": cfg.experiment_dataset,
        "metadataset_run_id": getattr(run1, "run_id", None),
        "materialized_run_id": getattr(run2, "run_id", None),
    }
    pointers_path = os.path.join(out_dir, "dataset_run_pointers.json")
    write_json(pointers_path, pointers)

    print("Created dataset runs.")
    print("Run1 (metadataset):", pointers["metadataset_run_id"])
    print("Run2 (materialized):", pointers["materialized_run_id"])
    print("Pointers saved to:", pointers_path)

    # Log the pointer file as an artifact to the *second* run for convenience.
    run2.log_artifact(local_path=pointers_path, artifact_path="dataset")

if __name__ == "__main__":
    main()
