"""Train CatBoost models (GPU-ready) on pre-generated NumPy datasets."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from catboost import CatBoostClassifier, CatBoostError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from metrics.score import normalized_rooms_score
    print("Successfully imported 'normalized_rooms_score'.")
except ImportError:
    print("Warning: 'normalized_rooms_score' not found. Using fallback metric.")

    def normalized_rooms_score(preds: np.ndarray, targets: np.ndarray) -> float:
        tp = (preds * targets).sum(axis=1)
        fp = (preds * (1 - targets)).sum(axis=1)
        fn = ((1 - preds) * targets).sum(axis=1)
        safe_denominator = np.maximum(tp + fp + fn, 1)
        return float(np.mean(tp / safe_denominator))

DATA_DIR = Path(os.environ.get("HACKATHON_DATA_DIR", Path(__file__).resolve().parent / "generated_datasets"))
MODEL_DIR = Path(os.environ.get("HACKATHON_MODEL_DIR", Path(__file__).resolve().parent / "catboost_models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
    split_path = DATA_DIR / f"{split}_dataset.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split '{split}' not found at {split_path}.")

    with np.load(split_path) as payload:
        if "X" not in payload.files or "y" not in payload.files:
            raise KeyError(f"File {split_path} must contain 'X' and 'y' arrays.")
        X = payload["X"].astype(np.float32)
        y = payload["y"].astype(np.float32)
    return X, y


def _build_catboost_params() -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": int(os.environ.get("CATBOOST_ITERATIONS", 2000)),
        "learning_rate": float(os.environ.get("CATBOOST_LR", 0.05)),
        "depth": int(os.environ.get("CATBOOST_DEPTH", 6)),
        "l2_leaf_reg": float(os.environ.get("CATBOOST_L2", 3.0)),
        "random_seed": int(os.environ.get("CATBOOST_SEED", 42)),
        "use_best_model": True,
        "task_type": os.environ.get("CATBOOST_TASK_TYPE", "GPU"),
        "devices": os.environ.get("CATBOOST_DEVICES", "0"),
        "allow_writing_files": False,
        "od_type": "Iter",
        "od_wait": int(os.environ.get("CATBOOST_EARLY_STOP", 200)),
        "verbose": False,
    }


def _fit_single_label(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
) -> CatBoostClassifier:
    model = CatBoostClassifier(**params)
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=False,
            early_stopping_rounds=params["od_wait"],
        )
        return model
    except CatBoostError:
        if params.get("task_type", "CPU").upper() != "GPU":
            raise
        print("GPU training failed. Falling back to CPU for this label.")
        cpu_params = dict(params)
        cpu_params["task_type"] = "CPU"
        cpu_params.pop("devices", None)
        cpu_model = CatBoostClassifier(**cpu_params)
        cpu_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=False,
            early_stopping_rounds=cpu_params["od_wait"],
        )
        return cpu_model


def _smoke_test_model_save(reference_model_path: Path) -> None:
    if not reference_model_path.exists():
        print("Skipping model save smoke test: reference model not found.")
        return

    probe = CatBoostClassifier()
    probe.load_model(reference_model_path)

    temp_path = reference_model_path.with_name(reference_model_path.stem + "_save_test.cbm")
    probe.save_model(temp_path)
    probe.load_model(temp_path)
    temp_path.unlink(missing_ok=True)
    print(f"Model save/reload smoke test succeeded via {reference_model_path}.")


def main() -> None:
    print(f"Loading NumPy datasets from {DATA_DIR.resolve()}...")
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")

    print(
        f"Train shape: X={X_train.shape}, y={y_train.shape} | "
        f"Val shape: X={X_val.shape}, y={y_val.shape}"
    )

    num_labels = y_train.shape[1]
    catboost_params = _build_catboost_params()
    print(f"CatBoost params: {catboost_params}")

    y_val_pred_matrix = np.zeros_like(y_val, dtype=np.float32)

    for label_idx in range(num_labels):
        print(f"--- Training label {label_idx + 1}/{num_labels} ---")
        y_train_i = y_train[:, label_idx]
        y_val_i = y_val[:, label_idx]

        model = _fit_single_label(X_train, y_train_i, X_val, y_val_i, catboost_params)
        y_val_pred_matrix[:, label_idx] = model.predict_proba(X_val)[:, 1].astype(np.float32)

        model_path = MODEL_DIR / f"catboost_model_label_{label_idx}.cbm"
        model.save_model(model_path)

    print(f"All models saved to {MODEL_DIR.resolve()}.")

    reference_model = MODEL_DIR / "catboost_model_label_0.cbm"
    _smoke_test_model_save(reference_model)

    y_val_binary = (y_val_pred_matrix > 0.5).astype(np.int32)
    final_score = normalized_rooms_score(y_val_binary, y_val)
    print("\n--- FINAL VALIDATION RESULTS ---")
    print(f"Custom Room Score: {final_score:.4f}")


if __name__ == "__main__":
    main()
