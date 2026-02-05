#!/usr/bin/env python3
"""
scripts/30_promote.py

Promotion step (curation):
- Reads runs from the training experiment
- Selects candidates (threshold and/or explicit promote flag)
- Creates model registry entry + version
- Writes provenance: source_run_ref, artifact path, sha, promoted_utc
- Avoids duplicate promotions (best-effort)

Env controls:
- PROMOTE_MIN_ACCURACY (default 0.98)
- PROMOTE_LIMIT        (default 50)   number of recent runs to inspect
- PROMOTE_BASE_VERSION (default 0.1.0)
"""

import os
from datetime import datetime, UTC

from util import load_config, sha256_file
from gitlab_mlops import Client


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
    return slug or "run"


def semver_version_for_promotion(run_name: str, run_ref: str, base: str) -> str:
    """
    Deterministic-ish and traceable version:
    - base is SemVer core (e.g. 0.1.0)
    - build metadata includes timestamp + run name + shortened run ref
    """
    short_ref = (run_ref or "unknown")[:12]
    return f"{base}+{utc_compact()}.{safe_slug(run_name)}.{safe_slug(short_ref)}"


def get_run_ref(run) -> str:
    return (
        getattr(run, "run_id", None)
        or getattr(run, "external_id", None)
        or getattr(run, "id", None)
        or "unknown"
    )


def get_run_param(run, key: str, default=None):
    """
    Best-effort extraction across client variants.
    """
    # common patterns: run.params dict, run.get_param(...)
    if hasattr(run, "get_param"):
        try:
            v = run.get_param(key)
            return v if v is not None else default
        except Exception:
            pass
    if hasattr(run, "params") and isinstance(run.params, dict):
        return run.params.get(key, default)
    # sometimes run.data.params
    if hasattr(run, "data") and hasattr(run.data, "params"):
        try:
            return run.data.params.get(key, default)
        except Exception:
            pass
    return default


def get_run_metric(run, key: str, default=None):
    """
    Best-effort metric extraction across client variants.
    """
    if hasattr(run, "get_metric"):
        try:
            v = run.get_metric(key)
            return v if v is not None else default
        except Exception:
            pass
    if hasattr(run, "metrics") and isinstance(run.metrics, dict):
        return run.metrics.get(key, default)
    if hasattr(run, "data") and hasattr(run.data, "metrics"):
        try:
            return run.data.metrics.get(key, default)
        except Exception:
            pass
    return default


def mark_run_promoted(run, model_name: str, version: str):
    """
    Best effort: some backends allow editing run params/tags after completion, some don't.
    """
    try:
        run.log_param("promotion_status", "promoted")
        run.log_param("promoted_model_name", model_name)
        run.log_param("promoted_model_version", version)
        run.log_param("promoted_utc", utc_iso_z())
    except Exception:
        pass

    if hasattr(run, "set_tag"):
        try:
            run.set_tag("promotion_status", "promoted")
            run.set_tag("promoted_model_version", version)
        except Exception:
            pass


def is_truthy(x) -> bool:
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


# -------------------------
# promotion
# -------------------------

def promote_run_to_model_registry(client: Client, model_name: str, run, model_path: str, version: str):
    """
    Creates/gets the Model Registry model and creates a version.
    Stores lineage/provenance on the model version.
    """
    model = client.get_model(name=model_name)
    if not model:
        model = client.create_model(
            name=model_name,
            description="MNIST CNN demo model. Curated versions are promoted from training runs.",
        )

    run_ref = get_run_ref(run)

    # NOTE: your earlier paste had a syntax error "create_version(a ...)" — fixed here.
    mv = model.create_version(
        description=f"Promoted from training run {run_ref}",
        version=version,
    )

    mv.log_param("source_experiment", "mnist_training")
    mv.log_param("source_run_ref", run_ref)
    mv.log_param("run_name", get_run_param(run, "run_name", default="unknown"))
    mv.log_param("artifact_path", get_run_param(run, "artifact_path_model_state", default="model/model_state_dict.pt"))

    # For reproducibility / dedupe:
    mv.log_param("model_state_sha256", sha256_file(model_path))
    mv.log_param("promoted_utc", utc_iso_z())

    mv.log_text(
        "This model version was curated and promoted by scripts/30_promote.py.\n"
        "Training artifacts live on the source training run; this model registry entry stores provenance.\n",
        "README.md",
    )

    return mv


def main():
    cfg = load_config()
    client = Client(tracking_uri=cfg.tracking_uri, gitlab_token=cfg.token)

    training_exp = client.get_experiment(name=cfg.experiment_training) or client.create_experiment(cfg.experiment_training)

    min_acc = float(os.environ.get("PROMOTE_MIN_ACCURACY", "0.98"))
    limit = int(os.environ.get("PROMOTE_LIMIT", "50"))
    base_version = os.environ.get("PROMOTE_BASE_VERSION", "0.1.0")

    # How to list runs depends on client. Common patterns:
    # - training_exp.list_runs()
    # - training_exp.search_runs()
    # We'll try a few.
    runs = []
    if hasattr(training_exp, "list_runs"):
        runs = training_exp.list_runs(max_results=limit)  # type: ignore
    elif hasattr(training_exp, "search_runs"):
        runs = training_exp.search_runs(max_results=limit)  # type: ignore
    elif hasattr(training_exp, "get_runs"):
        runs = training_exp.get_runs(max_results=limit)  # type: ignore
    else:
        raise RuntimeError("Training experiment object does not support listing runs (list_runs/search_runs/get_runs).")

    promoted = []
    inspected = 0

    for run in runs:
        inspected += 1
        run_ref = get_run_ref(run)
        run_name = get_run_param(run, "run_name", default="unknown")

        # Skip if already promoted (best effort)
        promo_status = get_run_param(run, "promotion_status", default=None)
        if str(promo_status).strip().lower() == "promoted":
            continue

        # Manual promotion flag
        promote_flag = (
            get_run_param(run, "promote", default=None)
            or get_run_param(run, "promotion_candidate", default=None)
        )
        manual = is_truthy(promote_flag)

        # Metric threshold
        acc = get_run_metric(run, "test_accuracy", default=None)
        try:
            acc_f = float(acc) if acc is not None else None
        except Exception:
            acc_f = None

        passes_threshold = (acc_f is not None) and (acc_f >= min_acc)

        if not (manual or passes_threshold):
            continue

        # Locate local model artifact if present (this promotion script assumes the artifact exists locally
        # in the same CI workspace). If you want to download from tracking storage, we can extend this.
        #
        # Training script writes to outputs/models/<run_name>_<timestamp>/model_state_dict.pt but that is not
        # persisted across jobs unless you keep it as CI artifacts. Best practice in CI: make outputs/ a job artifact.
        #
        # For now, we rely on an env var override or a convention:
        local_model_path = os.environ.get("PROMOTE_MODEL_PATH")
        if not local_model_path:
            # best-effort: if training job artifact is present, user can pass it in;
            # otherwise this will raise.
            raise RuntimeError(
                "PROMOTE_MODEL_PATH is not set. Provide the local path to model_state_dict.pt "
                "for the run you are promoting (e.g., from CI job artifacts)."
            )

        version = semver_version_for_promotion(run_name=run_name, run_ref=run_ref, base=base_version)

        mv = promote_run_to_model_registry(
            client=client,
            model_name=cfg.model_name,
            run=run,
            model_path=local_model_path,
            version=version,
        )

        mark_run_promoted(run, cfg.model_name, version)

        promoted.append(
            {
                "run_ref": run_ref,
                "run_name": run_name,
                "test_accuracy": acc_f,
                "model_name": cfg.model_name,
                "model_version": getattr(mv, "version", version),
            }
        )

    print(f"Inspected {inspected} runs. Promoted {len(promoted)}.")
    for p in promoted:
        print(p)


if __name__ == "__main__":
    main()
