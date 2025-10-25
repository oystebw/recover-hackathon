"""Builds CatBoost-friendly datasets with extended feature vectors.

This script augments the base 388 work-operation features with:
1. An OR-combined vector (length 388) that aggregates every job seen in the
   calculus rooms linked to the current sample.
2. An 11-dimensional room indicator that flags the current room as well as any
   rooms mentioned inside the calculus context.

The resulting feature tensor therefore has 388 + 388 + 11 = 787 columns.
Each split (train/val/test by default) is serialized to a .pt file containing
the features, labels, and ids along with a small metadata JSON file. The script
now emits multiple variants (low_hidden, baseline, high_hidden) so we can train
models under different probabilities of hiding work operations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.work_operations import WorkOperationsDataset  # noqa: E402


DEFAULT_DATA_ROOT = "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "precomputed" / "catboost_features"
DEFAULT_SEED = 42
DEFAULT_SPLITS = ("train", "val", "test")


VARIANT_CONFIGS: dict[str, dict[str, object]] = {
    "low_hidden": {
        "description": "Reduced probability that jobs are hidden (smaller sample_pct)",
        "sampling_strategy": [
            {
                "subset_size": 0.5,
                "sample_pct": 0.35,
                "use_balanced_data": True,
                "use_sampled_calculus": True,
            },
            {
                "subset_size": 0.5,
                "sample_pct": 0.2,
                "use_balanced_data": False,
                "use_sampled_calculus": True,
            },
        ],
    },
    "baseline": {
        "description": "Original sampling configuration",
        "sampling_strategy": None,
    },
    "high_hidden": {
        "description": "Increased probability that jobs are hidden (larger sample_pct)",
        "sampling_strategy": [
            {
                "subset_size": 0.5,
                "sample_pct": 0.65,
                "use_balanced_data": True,
                "use_sampled_calculus": True,
            },
            {
                "subset_size": 0.5,
                "sample_pct": 0.45,
                "use_balanced_data": False,
                "use_sampled_calculus": True,
            },
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Path to raw CSV data folder")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the .pt files and metadata will be stored",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=sorted(DEFAULT_SPLITS),
        help="Dataset splits to build",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed forwarded to the dataset")
    return parser.parse_args()


def _or_merge_calculus(calculus: Iterable[dict], num_clusters: int) -> torch.Tensor:
    """Return a vector marking any job performed inside the calculus rooms."""

    aggregated = torch.zeros(num_clusters, dtype=torch.float32)
    for ctx in calculus:
        aggregated = torch.maximum(aggregated, ctx["work_operations_index_encoded"].to(torch.float32))
    return aggregated


def _room_indicator(sample: dict, room_to_index: dict[str, int]) -> torch.Tensor:
    """Flag the target room and all rooms referenced in the calculus list."""

    indicator = torch.zeros(len(room_to_index), dtype=torch.float32)

    def _flag(room_name: str | None) -> None:
        if room_name is None:
            return
        idx = room_to_index.get(room_name)
        if idx is not None:
            indicator[idx] = 1.0

    _flag(sample.get("room_cluster"))
    for ctx in sample.get("calculus", []):
        _flag(ctx.get("room_cluster"))
    return indicator


def build_feature_vector(sample: dict, num_clusters: int, room_to_index: dict[str, int]) -> torch.Tensor:
    base_x = sample["X"].to(torch.float32)
    calculus_or = _or_merge_calculus(sample.get("calculus", []), num_clusters)
    room_vector = _room_indicator(sample, room_to_index)
    return torch.cat([base_x, calculus_or, room_vector], dim=0)


def collect_split(
    dataset: WorkOperationsDataset, room_to_index: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    ids: list[int] = []
    single_label_count = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        features.append(build_feature_vector(sample, dataset.num_clusters, room_to_index))

        label = sample["Y"].to(torch.float32)
        labels.append(label)
        ids.append(int(sample["id"]))

        if int(label.sum().item()) == 1:
            single_label_count += 1

    stacked_features = torch.stack(features)
    stacked_labels = torch.stack(labels)
    stacked_ids = torch.tensor(ids, dtype=torch.int64)
    return stacked_features, stacked_labels, stacked_ids, single_label_count


def save_split(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, path)


def inject_zero_label_mass(labels: torch.Tensor, value: float = 0.01) -> tuple[torch.Tensor, list[int]]:
    """Ensure every label column has at least one positive entry."""

    column_sums = labels.sum(dim=0)
    zero_columns = (column_sums == 0).nonzero(as_tuple=False).squeeze(-1)
    if zero_columns.numel() == 0:
        return labels, []

    adjusted = labels.clone()
    target_row = 0  # arbitrary sample to hold the synthetic signal
    for col in zero_columns.tolist():
        adjusted[target_row, col] = value
    return adjusted, zero_columns.tolist()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant_name, variant_cfg in VARIANT_CONFIGS.items():
        print(f"\n=== Building variant: {variant_name} ===")
        variant_dir = output_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        sampling_strategy = variant_cfg.get("sampling_strategy")
        splits: dict[str, WorkOperationsDataset] = {}
        for split in args.splits:
            strategy = sampling_strategy if (sampling_strategy is not None and split != "test") else None
            splits[split] = WorkOperationsDataset(
                root=args.data_root,
                split=split,
                download=False,
                seed=args.seed,
                sampling_strategy=strategy,
            )
            if split == "test" and sampling_strategy is not None:
                print("[info] Test split always uses default sampling (no hiding).")

        train_dataset = splits["train"]
        room_to_index = train_dataset.room_to_index
        num_rooms = len(room_to_index)
        feature_dim = train_dataset.num_clusters * 2 + num_rooms

        metadata = {
            "feature_dim": feature_dim,
            "num_clusters": train_dataset.num_clusters,
            "num_rooms": num_rooms,
            "room_order": list(room_to_index.keys()),
            "splits": args.splits,
            "description": "Base X (388) || calculus OR (388) || room indicator (11)",
            "variant": variant_name,
            "variant_description": variant_cfg.get("description", ""),
            "sampling_strategy": sampling_strategy,
        }
        with (variant_dir / "metadata.json").open("w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, ensure_ascii=False)

        for split_name in args.splits:
            dataset = splits[split_name]
            features, labels, ids, single_label_count = collect_split(dataset, room_to_index)

            injected_columns: list[int] = []
            if split_name == "train":
                labels, injected_columns = inject_zero_label_mass(labels)
                if injected_columns:
                    print(
                        f"[{variant_name}:{split_name}] injected {len(injected_columns)} zero-sum label columns with value 0.01"
                    )

            print(
                f"[{variant_name}:{split_name}] features shape: {tuple(features.shape)}, labels shape: {tuple(labels.shape)}"
            )
            print(
                f"[{variant_name}:{split_name}] samples with a single positive label: {single_label_count} / {labels.shape[0]}"
            )
            if injected_columns:
                print(f"[{variant_name}:{split_name}] zero-sum label indices: {injected_columns}")

            save_split(
                {"features": features, "labels": labels, "ids": ids},
                variant_dir / f"catboost_{split_name}.pt",
            )


if __name__ == "__main__":
    main()
