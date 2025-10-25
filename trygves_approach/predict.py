"""Generate Kaggle submission using GPU-trained CatBoost models and NumPy datasets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = Path(os.environ.get("HACKATHON_DATA_DIR", Path(__file__).resolve().parent / "generated_datasets"))
MODEL_DIR = Path(os.environ.get("HACKATHON_MODEL_DIR", Path(__file__).resolve().parent / "catboost_models"))
SUBMISSION_FILENAME = os.environ.get("SUBMISSION_FILENAME", "submission.csv")
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", 0.5))


def _load_test_split() -> tuple[np.ndarray, np.ndarray, int]:
    test_path = DATA_DIR / "test_dataset.npz"
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}.")

    with np.load(test_path) as payload:
        if "X" not in payload.files:
            raise KeyError(f"File {test_path} must contain an 'X' array.")
        X = payload["X"].astype(np.float32)
        ids = payload["ids"].astype(np.int64) if "ids" in payload.files else np.arange(X.shape[0], dtype=np.int64)
        num_submission_rows = (
            int(payload["num_submission_rows"]) if "num_submission_rows" in payload.files else int(ids.max() + 1)
        )
    return X, ids, num_submission_rows


def _discover_num_labels() -> int:
    explicit = os.environ.get("NUM_LABELS")
    if explicit:
        return int(explicit)

    indices: list[int] = []
    for model_path in MODEL_DIR.glob("catboost_model_label_*.cbm"):
        try:
            indices.append(int(model_path.stem.split("_")[-1]))
        except ValueError:
            continue

    if not indices:
        raise FileNotFoundError(f"No CatBoost model files found under {MODEL_DIR}.")

    return max(indices) + 1


def _load_models(num_labels: int) -> list[CatBoostClassifier]:
    models: list[CatBoostClassifier] = []
    for label_idx in range(num_labels):
        model_path = MODEL_DIR / f"catboost_model_label_{label_idx}.cbm"
        if not model_path.exists():
            raise FileNotFoundError(f"Expected model file missing: {model_path}")
        model = CatBoostClassifier()
        model.load_model(model_path)
        models.append(model)
    return models


def main() -> None:
    print(f"Loading test split from {DATA_DIR.resolve()}...")
    X_unseen, test_ids, num_submission_rows = _load_test_split()
    num_samples = X_unseen.shape[0]
    print(f"Loaded {num_samples} samples with feature dim {X_unseen.shape[1]}.")

    num_labels = _discover_num_labels()
    print(f"Expecting {num_labels} label models.")

    print(f"Loading CatBoost models from {MODEL_DIR.resolve()}...")
    models = _load_models(num_labels)
    print("All models loaded successfully.")

    y_pred_matrix = np.zeros((num_samples, num_labels), dtype=np.float32)
    for label_idx, model in enumerate(models):
        y_pred_matrix[:, label_idx] = model.predict_proba(X_unseen)[:, 1].astype(np.float32)

    y_pred_binary = (y_pred_matrix > PREDICTION_THRESHOLD).astype(np.int32)

    final_data = np.zeros((num_submission_rows, num_labels), dtype=np.int32)
    mapped = 0
    for sample_idx, submission_id in enumerate(test_ids):
        if 0 <= submission_id < num_submission_rows:
            final_data[submission_id] = y_pred_binary[sample_idx]
            mapped += 1

    print(f"Mapped {mapped}/{num_samples} predictions to submission rows.")

    final_ids = np.arange(num_submission_rows, dtype=np.int64).reshape(-1, 1)
    submission_matrix = np.hstack((final_ids, final_data))
    header = ["id"] + [str(i) for i in range(num_labels)]
    df_submission = pd.DataFrame(submission_matrix, columns=header)
    df_submission = df_submission.astype(int)
    df_submission.to_csv(SUBMISSION_FILENAME, index=False)

    print("\n--- SUBMISSION FILE SAVED ---")
    print(f"Wrote {num_submission_rows} rows to {SUBMISSION_FILENAME}")


if __name__ == "__main__":
    main()
