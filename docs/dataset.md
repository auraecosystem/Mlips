# Dataset curation

This demo logs two kinds of datasets into GitLab-backed MLflow:

1) **Metadataset**: a JSONL manifest describing samples (labels, split, checksums, etc).
   - Cheap to store
   - Lets you iterate on labeling / metadata without moving big blobs
   - Good for governance + reproducibility

2) **Materialized dataset artifact**: a small, real dataset snapshot logged as artifacts.
   - Useful when you need frozen bytes for reproducible training
   - Useful for off-cluster curation + on-cluster training

See `scripts/10_create_datasets.py`.
