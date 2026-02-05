import os
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import mlflow

from util import load_config, ensure_dir


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


def download_model_state(run_id: str, dst_dir: str) -> str:
    # This uses the standard MLflow artifact download.
    # It works as long as MLFLOW_TRACKING_URI/TOKEN are configured to point at GitLab's MLflow API endpoint.
    ensure_dir(dst_dir)
    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/model_state_dict.pt", dst_path=dst_dir)
    return local_path


def main():
    cfg = load_config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True, help="Training run_id to fetch model artifact from (MLflow).")
    ap.add_argument("--n", type=int, default=16, help="Number of test samples to run inference on.")
    args = ap.parse_args()

    # Configure MLflow for artifact download
    os.environ["MLFLOW_TRACKING_URI"] = cfg.tracking_uri
    os.environ["MLFLOW_TRACKING_TOKEN"] = cfg.token

    model_state = download_model_state(args.run_id, dst_dir=os.path.join("outputs", "infer", args.run_id))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleCNN().to(device)
    sd = torch.load(model_state, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    tfm = transforms.Compose([transforms.ToTensor()])
    test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

    correct = 0
    for i in range(min(args.n, len(test_ds))):
        x, y = test_ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(dim=1).item())
        ok = (pred == int(y))
        correct += 1 if ok else 0
        print(f"idx={i} pred={pred} label={int(y)} ok={ok}")

    acc = correct / max(min(args.n, len(test_ds)), 1)
    print(f"Accuracy on {args.n} samples: {acc:.3f}")

if __name__ == "__main__":
    main()
