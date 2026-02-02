# Troubleshooting

- **Auth errors / 401**: verify `MLFLOW_TRACKING_TOKEN` and the token's permissions.
- **Wrong tracking URI**: ensure `MLFLOW_TRACKING_URI` targets your project id.
- **SSL / corporate proxy issues**: try `REQUESTS_CA_BUNDLE` or your org's CA chain.
- **CUDA issues**: this demo runs fine on CPU; CUDA is optional.

Tip: run `python scripts/00_check_env.py` to validate environment variables.
