"""Build per-room job availability metadata from the baseline CatBoost dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "precomputed" / "catboost_features" / "baseline"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "precomputed" / "room_clusters"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="Directory that contains catboost_train.pt and metadata.json for the baseline dataset.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root folder that stores the original CSV splits (used to resolve room assignments).",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Split to analyze.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the derived room-cluster dataset will be saved.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed forwarded to WorkOperationsDataset.")
    return parser.parse_args()


def _load_baseline_split(baseline_dir: Path, split: str) -> dict:
    split_path = baseline_dir / f"catboost_{split}.pt"
    if not split_path.exists():
        raise FileNotFoundError(f"Could not locate {split_path}")
    return torch.load(split_path, map_location="cpu")


def _load_metadata(baseline_dir: Path) -> dict:
    meta_path = baseline_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Baseline metadata missing at {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _room_assignments(data_root: Path, split: str, seed: int) -> tuple[Dict[int, int], Dict[int, str]]:
    from dataset.work_operations import WorkOperationsDataset  # local import to avoid heavy dependency at module import

    dataset = WorkOperationsDataset(root=data_root, split=split, seed=seed, download=False)
    room_to_index = dataset.room_to_index
    index_to_room = {idx: name for name, idx in room_to_index.items()}

    mapping: Dict[int, int] = {}
    unknown_counts: Dict[str, int] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        room_name = sample["room_cluster"]
        room_idx = room_to_index.get(room_name)
        if room_idx is None:
            unknown_counts[room_name] = unknown_counts.get(room_name, 0) + 1
            continue
        mapping[int(sample["id"])] = room_idx

    if unknown_counts:
        print(
            "Warning: skipping samples with unknown rooms -> "
            + ", ".join(f"{name}:{count}" for name, count in unknown_counts.items())
        )

    return mapping, index_to_room


def build_room_job_stats(
    payload: dict,
    room_lookup: Dict[int, int],
    index_to_room: Dict[int, str],
) -> dict:
    features: torch.Tensor = payload["features"].to(torch.float32)
    labels: torch.Tensor = payload["labels"].to(torch.float32)
    ids: torch.Tensor = payload["ids"].to(torch.int64)

    num_jobs = labels.shape[1]
    base_features = features[:, :num_jobs]
    base_features_np = base_features.numpy()
    labels_np = labels.numpy()

    room_count = len(index_to_room)
    job_masks = np.zeros((room_count, num_jobs), dtype=bool)
    sample_indices: Dict[int, List[int]] = {idx: [] for idx in range(room_count)}

    missing_rooms: set[int] = set()

    for row_idx, sample_id in enumerate(ids.tolist()):
        room_idx = room_lookup.get(int(sample_id))
        if room_idx is None:
            missing_rooms.add(int(sample_id))
            continue

        mask = np.logical_or(base_features_np[row_idx] > 0.5, labels_np[row_idx] > 0.5)
        job_masks[room_idx] |= mask
        sample_indices[room_idx].append(row_idx)

    if missing_rooms:
        print(f"Warning: {len(missing_rooms)} samples had no room assignment (ids: {sorted(list(missing_rooms))[:5]} ...)")

    room_entries = []
    for room_idx in range(room_count):
        mask = job_masks[room_idx]
        ever_jobs = np.where(mask)[0].tolist()
        never_jobs = np.where(~mask)[0].tolist()
        room_entries.append(
            {
                "room_index": room_idx,
                "room_name": index_to_room.get(room_idx, f"room_{room_idx}"),
                "num_samples": len(sample_indices[room_idx]),
                "ever_jobs": ever_jobs,
                "never_jobs": never_jobs,
                "sample_indices": sample_indices[room_idx],
            }
        )

    return {
        "room_entries": room_entries,
        "num_jobs": num_jobs,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(args.baseline_dir)
    payload = _load_baseline_split(args.baseline_dir, args.split)
    room_lookup, index_to_room = _room_assignments(args.data_root, args.split, args.seed)

    stats = build_room_job_stats(payload, room_lookup, index_to_room)

    artifact = {
        "split": args.split,
        "baseline_dir": str(args.baseline_dir.resolve()),
        "num_jobs": stats["num_jobs"],
        "room_order": metadata.get("room_order", [index_to_room[idx] for idx in sorted(index_to_room.keys())]),
        "room_clusters": stats["room_entries"],
    }

    output_pt = args.output_dir / f"room_cluster_dataset_{args.split}.pt"
    torch.save(artifact, output_pt)

    summary = [
        {
            "room_index": entry["room_index"],
            "room_name": entry["room_name"],
            "num_samples": entry["num_samples"],
            "ever_jobs": len(entry["ever_jobs"]),
            "never_jobs": len(entry["never_jobs"]),
        }
        for entry in artifact["room_clusters"]
    ]
    summary_path = args.output_dir / f"room_cluster_summary_{args.split}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved room-cluster dataset -> {output_pt}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
