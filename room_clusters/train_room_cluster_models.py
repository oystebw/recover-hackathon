"""Train room-specific CatBoost clusters that only model jobs previously observed in that room."""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from catboost import CatBoostClassifier, Pool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOM_CLUSTER_DATASET = PROJECT_ROOT / "precomputed" / "room_clusters" / "room_cluster_dataset_train.pt"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "room_cluster_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--room-cluster-dataset",
        type=Path,
        default=DEFAULT_ROOM_CLUSTER_DATASET,
        help="Path to the *.pt artifact produced by build_room_cluster_dataset.py",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Optional override for the baseline CatBoost directory (otherwise inferred from the artifact).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory where room-cluster model folders will be created.",
    )
    parser.add_argument(
        "--room",
        action="append",
        default=None,
        help="Limit training to the given room name (can be provided multiple times).",
    )
    parser.add_argument(
        "--room-index",
        type=int,
        action="append",
        default=None,
        help="Limit training to the given room index (0-based).",
    )
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction within each room subset.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for data shuffling.")
    parser.add_argument("--overwrite", action="store_true", help="Retrain and overwrite existing models.")
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-iter", type=int, default=1500)
    parser.add_argument("--catboost-lr", type=float, default=0.05)
    return parser.parse_args()


def _load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Room-cluster dataset not found at {path}")
    return torch.load(path, map_location="cpu")


def _load_baseline(payload_dir: Path, split: str) -> dict:
    payload_path = payload_dir / f"catboost_{split}.pt"
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing baseline payload at {payload_path}")
    return torch.load(payload_path, map_location="cpu")


def _catboost_params(args: argparse.Namespace) -> dict:
    return {
        "loss_function": "Logloss",
        "depth": args.catboost_depth,
        "learning_rate": args.catboost_lr,
        "iterations": args.catboost_iter,
        "l2_leaf_reg": 5.0,
        "random_seed": args.seed,
        "thread_count": -1,
        "eval_metric": "AUC",
        "allow_writing_files": False,
        "task_type": "CPU",
        "class_weights": [1.25, 1.5],
        "verbose": 100,
    }


def _slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    slug = "".join(ch if ch.isalnum() else "_" for ch in ascii_name.lower()).strip("_")
    return slug or "room"


def _train_val_split(num_samples: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_size = int(math.floor(num_samples * val_frac))
    val_size = min(max(val_size, 1 if num_samples > 1 else 0), num_samples - 1 if num_samples > 1 else 0)
    if val_size == 0:
        return indices, np.array([], dtype=np.int64)
    return indices[val_size:], indices[:val_size]


def _select_rooms(artifact: dict, args: argparse.Namespace) -> list[dict]:
    rooms = artifact["room_clusters"]
    if not args.room and not args.room_index:
        return rooms

    selected = []
    room_names = {entry["room_name"].lower(): entry for entry in rooms}
    if args.room:
        for name in args.room:
            key = name.lower()
            if key not in room_names:
                raise ValueError(f"Unknown room name '{name}'")
            selected.append(room_names[key])

    if args.room_index:
        index_map = {entry["room_index"]: entry for entry in rooms}
        for idx in args.room_index:
            if idx not in index_map:
                raise ValueError(f"Unknown room index {idx}")
            selected.append(index_map[idx])

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for entry in selected:
        key = (entry["room_index"], entry["room_name"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _ensure_room_dir(root: Path, entry: dict) -> Path:
    slug = _slugify(entry["room_name"])
    room_dir = root / f"room_{entry['room_index']:02d}_{slug}"
    room_dir.mkdir(parents=True, exist_ok=True)
    return room_dir


def _save_room_metadata(
    room_dir: Path,
    entry: dict,
    trained_jobs: Iterable[int],
    forced_zero_jobs: Iterable[int],
    forced_one_jobs: Iterable[int],
) -> None:
    zero_jobs = set(entry["never_jobs"])
    zero_jobs.update(forced_zero_jobs)
    metadata = {
        "room_index": entry["room_index"],
        "room_name": entry["room_name"],
        "num_samples": entry["num_samples"],
        "ever_jobs": entry["ever_jobs"],
        "never_jobs": entry["never_jobs"],
        "trained_jobs": list(trained_jobs),
        "forced_zero_jobs": sorted(zero_jobs),
        "forced_one_jobs": sorted(set(forced_one_jobs)),
    }
    (room_dir / "cluster_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def train_room_cluster(
    entry: dict,
    base_features: np.ndarray,
    labels: np.ndarray,
    params: dict,
    args: argparse.Namespace,
) -> None:
    if not entry["sample_indices"]:
        print(f"[room {entry['room_name']}] no samples available, skipping")
        return

    allowed_jobs = entry["ever_jobs"]
    if not allowed_jobs:
        print(f"[room {entry['room_name']}] no observed jobs, skipping")
        return

    sample_idx = np.array(entry["sample_indices"], dtype=np.int64)
    X_room = base_features[sample_idx][:, allowed_jobs].astype(np.float32)
    y_room = labels[sample_idx]

    print(
        f"[room {entry['room_name']}] cluster jobs={len(allowed_jobs)} | input_dim={X_room.shape[1]} | samples={len(sample_idx)}"
    )

    base_train_idx, base_val_idx = _train_val_split(len(sample_idx), args.val_frac, args.seed)
    base_has_val = base_val_idx.size > 0

    room_dir = _ensure_room_dir(args.model_dir, entry)
    trained_jobs: list[int] = []
    forced_zero_jobs: list[int] = []
    forced_one_jobs: list[int] = []

    for job_id in allowed_jobs:
        train_idx = base_train_idx.copy()
        val_idx = base_val_idx.copy()
        has_val = base_has_val
        model_path = room_dir / f"catboost_job_{job_id}.cbm"
        if model_path.exists() and not args.overwrite:
            trained_jobs.append(job_id)
            continue

        target = y_room[:, job_id].astype(np.float32)
        unique_vals = np.unique(target)
        if unique_vals.size == 1:
            const_val = unique_vals[0]
            if const_val <= 0:
                print(
                    f"[room {entry['room_name']}] job {job_id} never hidden (Y constant 0); forcing deterministic zero"
                )
                forced_zero_jobs.append(job_id)
            else:
                print(
                    f"[room {entry['room_name']}] job {job_id} always hidden (Y constant 1); forcing deterministic one"
                )
                forced_one_jobs.append(job_id)
            continue

        if train_idx.size == 0:
            train_idx = np.arange(len(sample_idx))
            val_idx = np.array([], dtype=np.int64)
            has_val = False

        unique_train = np.unique(target[train_idx])
        if unique_train.size < 2:
            full_unique = np.unique(target)
            if full_unique.size < 2:
                if full_unique[0] <= 0:
                    print(
                        f"[room {entry['room_name']}] job {job_id} target degenerate; forcing deterministic zero"
                    )
                    forced_zero_jobs.append(job_id)
                else:
                    print(
                        f"[room {entry['room_name']}] job {job_id} target degenerate; forcing deterministic one"
                    )
                    forced_one_jobs.append(job_id)
                continue
            train_idx = np.arange(len(sample_idx))
            val_idx = np.array([], dtype=np.int64)
            has_val = False

        X_train = X_room[train_idx]
        y_train = target[train_idx]
        train_pool = Pool(X_train, y_train)

        eval_set = None
        if has_val:
            X_val = X_room[val_idx]
            y_val = target[val_idx]
            if np.unique(y_val).size == 1:
                eval_set = None
            else:
                eval_set = Pool(X_val, y_val)

        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=eval_set is not None,
            verbose=params.get("verbose", 100),
        )
        model.save_model(model_path)
        trained_jobs.append(job_id)
        print(f"[room {entry['room_name']}] trained job {job_id} -> {model_path}")

    remaining_jobs = (
        set(allowed_jobs)
        - set(trained_jobs)
        - set(forced_zero_jobs)
        - set(forced_one_jobs)
    )
    if remaining_jobs:
        print(
            f"[room {entry['room_name']}] warning: {len(remaining_jobs)} jobs unaccounted (check logs)."
        )

    _save_room_metadata(room_dir, entry, trained_jobs, forced_zero_jobs, forced_one_jobs)


def main() -> None:
    args = parse_args()
    artifact = _load_artifact(args.room_cluster_dataset)
    baseline_dir = args.baseline_dir or Path(artifact["baseline_dir"])
    args.model_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_baseline(baseline_dir, artifact["split"])
    num_jobs = artifact["num_jobs"]
    base_features = payload["features"].to(torch.float32)[:, :num_jobs].numpy()
    labels = payload["labels"].to(torch.float32).numpy()

    params = _catboost_params(args)
    rooms = _select_rooms(artifact, args)

    print(f"Preparing to train {len(rooms)} room clusters using baseline data from {baseline_dir}.")
    for entry in rooms:
        train_room_cluster(entry, base_features, labels, params, args)


if __name__ == "__main__":
    main()
