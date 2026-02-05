#!/usr/bin/env python3
"""
scripts/30_promote.py

Promotion step (curation) using MLflow for run discovery/artifacts and gitlab_mlops for Model Registry.

Flow:
1) Search runs in the training experiment via mlflow.search_runs
2) Pick candidates by threshold and/or explicit promote flag
3) Download the model artifact from the run using MlflowClient.download_artifacts
4) Create/get model registry entry + create model version
5) Log lineage/provenance on the model version
6) Mark the training run as promoted (tag)

Env controls:
- PROMOTE_MIN_ACCURACY   (default 0.98)
- PROMOTE_LIMIT          (default 50)   number of recent runs to inspect
- PROMOTE_BASE_VERSION   (default 0.1.0)
- PROMOTE_ARTIFACT_PATH  (default "model/model_state_dict.pt")
- PROMOTE_REQUIRE_FLAG   (default "false") if true, ONLY runs with promote=true are promoted
"""

import os
from datetime import datetime, UTC

import mlflow
from mlflow.tracking import MlflowClient

from util import load_config, sha256_file, ensure_dir
from gitlab_mlops import Client as GitLabMLOpsClient


# -------------------------
# helpers
# -------------------------

def utc_iso_z() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def utc_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_slug(s: str) -> str:
    s = (s or "").strip()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out).strip("-")
    return slug or "x"


def is_truthy(x) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def semver_version_for_promotion(run_name: str, run_id: str, base: str) -> str:
    short_id = (run_id or "unknown")[:12]
    return f"{base}+{utc_compact()}.{safe_slug(run_name)}.{safe_slug(short_id)}"


def get_field(row, *keys, default=None):
    """
    Helper for mlflow.search_runs DataFrame rows.
    Tries multiple keys (e.g., "params.run_name", "tags.run_name").
    """
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def mark_run_promoted(mlc: MlflowClient, run_id: str, model_name: str, version: str):
    # Best-effort tagging; supported by MLflow server implementations generally.
    try:
        mlc.set_tag(run_id, "promotion_status", "promoted")
        mlc.set_tag(run_id, "promoted_model_name", model_name)
        mlc.set_tag(run_id, "promoted_model_version", version)
        mlc.set_tag(run_id, "promoted_utc", utc_iso_z())
    except Exception:
        # Don’t fail promotion if tagging is blocked by backend policy
        pass


# -------------------------
# model registry promotion
# -------------------------

def promote_to_model_registry(
    gl_client: GitLabMLOpsClient,
    model_name: str,
    run_id: str,
    run_name: str,
    model_path_local: str,
    artifact_path_in_run: str,
    version: str,
    test_accuracy: float | None,
    model_sha: str | None,
):
    # Create or get model entry
    model = gl_client.get_model(name=model_name)
    if not model:
        model = gl_client.create_model(
            name=model_name,
            description="MNIST CNN demo model. Curated versions are promoted from training runs.",
        )

    mv = model.create_version(
        description=f"Promoted from training run {run_id}",
        version=version,
    )

    # Keep registry metadata focused on provenance + consumption
    mv.log_param("source_experiment", "mnist_training")
    mv.log_param("source_run_id", run_id)
    mv.log_param("run_name", run_name)
    mv.log_param("artifact_path", artifact_path_in_run)
    mv.log_param("promoted_utc", utc_iso_z())

    if test_accuracy is not None:
        mv.log_param("test_accuracy", float(test_accuracy))

    if model_sha is None:
        model_sha = sha256_file(model_path_local)
    mv.log_param("model_state_sha256", model_sha)

    mv.log_text(
        "This model version was curated and promoted by scripts/30_promote.py.\n"
        "Training artifacts live on the source training run; this registry entry stores provenance.\n",
        "README.md",
    )

    return mv


def main():
    cfg = load_config()

    # Ensure mlflow is configured (tracking uri/token are already env-driven by util.load_config expectations)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    # Token is usually picked up by the GitLab MLflow integration via env; keep as-is.

    mlc = MlflowClient()

    # For registry operations
    gl = GitLabMLOpsClient(tracking_uri=cfg.tracking_uri, gitlab_token=cfg.token)

    exp = mlflow.get_experiment_by_name(cfg.experiment_training)
    if exp is None:
        raise RuntimeError(f"Training experiment not found: {cfg.experiment_training}")

    min_acc = float(os.environ.get("PROMOTE_MIN_ACCURACY", "0.98"))
    limit = int(os.environ.get("PROMOTE_LIMIT", "50"))
    base_version = os.environ.get("PROMOTE_BASE_VERSION", "0.1.0")
    artifact_path = os.environ.get("PROMOTE_ARTIFACT_PATH", "model/model_state_dict.pt")
    require_flag = is_truthy(os.environ.get("PROMOTE_REQUIRE_FLAG", "false"))

    # Pull the most recent runs; MLflow returns a DataFrame.
    # Order by start_time desc so "most recent" promotions happen first.
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        max_results=limit,
    )

    inspected = 0
    promoted = []

    # Where we download artifacts
    ensure_dir("outputs")
    dl_root = os.path.join("outputs", "promotion_downloads")
    ensure_dir(dl_root)

    for _, row in df.iterrows():
        inspected += 1

        run_id = row["run_id"]
        run_name = get_field(row, "params.run_name", "tags.run_name", default="unknown")

        promo_status = get_field(row, "tags.promotion_status", "params.promotion_status", default=None)
        if str(promo_status).strip().lower() == "promoted":
            continue

        # Manual promote flag can be tag or param
        promote_flag = get_field(row, "tags.promote", "params.promote", "tags.promotion_candidate", "params.promotion_candidate", default=None)
        manual = is_truthy(promote_flag)

        acc = get_field(row, "metrics.test_accuracy", default=None)
        try:
            acc_f = float(acc) if acc is not None else None
        except Exception:
            acc_f = None

        passes_threshold = (acc_f is not None) and (acc_f >= min_acc)

        if require_flag:
            if not manual:
                continue
        else:
            if not (manual or passes_threshold):
                continue

        # Download artifact from MLflow/GitLab tracking storage
        dst_dir = os.path.join(dl_root, run_id)
        ensure_dir(dst_dir)
        try:
            local_path = mlc.download_artifacts(run_id, artifact_path, dst_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download artifact '{artifact_path}' for run_id={run_id}. "
                f"Ensure the artifact exists and the backend supports download_artifacts. Error: {e}"
            )

        # If artifact_path is a file, MLflow returns the local file path.
        # If it’s a directory, you'd need to locate the model file inside it.
        model_path_local = local_path
        if os.path.isdir(model_path_local):
            # Common case: user passed artifact dir, find the expected file name
            candidate = os.path.join(model_path_local, os.path.basename(artifact_path))
            if os.path.exists(candidate):
                model_path_local = candidate
            else:
                raise RuntimeError(
                    f"Downloaded artifact is a directory but expected model file not found. "
                    f"Downloaded to: {local_path}"
                )

        # Compute hash from the downloaded artifact
        model_sha = sha256_file(model_path_local)

        version = semver_version_for_promotion(run_name=run_name, run_id=run_id, base=base_version)

        mv = promote_to_model_registry(
            gl_client=gl,
            model_name=cfg.model_name,
            run_id=run_id,
            run_name=run_name,
            model_path_local=model_path_local,
            artifact_path_in_run=artifact_path,
            version=version,
            test_accuracy=acc_f,
            model_sha=model_sha,
        )

        # Mark run promoted (best effort)
        mark_run_promoted(mlc, run_id, cfg.model_name, version)

        promoted.append(
            {
                "run_id": run_id,
                "run_name": run_name,
                "test_accuracy": acc_f,
                "model_name": cfg.model_name,
                "model_version": getattr(mv, "version", version),
                "model_state_sha256": model_sha,
            }
        )

    print(f"Inspected {inspected} runs from experiment '{cfg.experiment_training}'. Promoted {len(promoted)}.")
    for p in promoted:
        print(p)


if __name__ == "__main__":
    main()
