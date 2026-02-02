from util import load_config

if __name__ == "__main__":
    cfg = load_config()
    print("OK: environment looks set.")
    print(f"MLFLOW_TRACKING_URI={cfg.tracking_uri}")
    print("TOKEN is set (not printing it).")
    print(f"Dataset experiment: {cfg.experiment_dataset}")
    print(f"Training experiment: {cfg.experiment_training}")
    print(f"Model name: {cfg.model_name}")
