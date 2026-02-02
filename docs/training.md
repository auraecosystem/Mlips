# Training + experiment tracking

Training uses PyTorch on a small MNIST subset to keep the demo fast.

- Two runs are produced with different hyperparameters.
- Each run logs:
  - parameters (lr, batch_size, epochs, etc)
  - metrics (loss, accuracy)
  - artifacts (model state_dict, run metadata)
  - references to the dataset run(s)

See `scripts/20_train_and_register.py`.
