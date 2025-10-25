"""Combine existing CatBoost feature variants into a single master dataset.

The script loads every requested split from the baseline, low-hidden, and
high-hidden variants under ``precomputed/catboost_features``. Each split is
concatenated along the sample dimension, shuffled, and saved to a new output
folder so downstream training jobs can rely on one consolidated dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "precomputed" / "catboost_features"
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR / "master"
DEFAULT_VARIANTS = ("low_hidden", "baseline", "high_hidden")
DEFAULT_SPLITS = ("train", "val", "test")
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory that stores the per-variant CatBoost feature folders",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Destination folder for the master dataset",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Variant folders to merge (order controls concatenation)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=sorted(DEFAULT_SPLITS),
        help="Dataset splits to process",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed used for shuffling")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing master split files",
    )
    return parser.parse_args()


def _torch_load(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return torch.load(path, map_location="cpu")


def _stack_payloads(payloads: Sequence[dict[str, torch.Tensor]], seed: int) -> dict[str, torch.Tensor]:
    features = torch.cat([p["features"] for p in payloads], dim=0)
    labels = torch.cat([p["labels"] for p in payloads], dim=0)
    ids = torch.cat([p["ids"] for p in payloads], dim=0)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)

    return {
        "features": features[permutation],
        "labels": labels[permutation],
        "ids": ids[permutation],
    }


def _load_base_metadata(source_dir: Path) -> dict:
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def _write_metadata(
    base_metadata: dict,
    output_dir: Path,
    variants: Sequence[str],
    split_sizes: dict[str, int],
    seed: int,
) -> None:
    metadata = dict(base_metadata)
    metadata.update(
        {
            "variant": "master",
            "variant_description": "Concatenation of low_hidden, baseline, and high_hidden variants",
            "source_variants": list(variants),
            "samples_per_split": split_sizes,
            "shuffle_seed": seed,
            "splits": list(split_sizes.keys()),
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _stable_split_seed(base_seed: int, split: str) -> int:
    offset = sum(split.encode("utf-8"))
    return base_seed + offset


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    variants = tuple(args.variants)
    splits = tuple(args.splits)

    output_dir.mkdir(parents=True, exist_ok=True)

    split_sizes: dict[str, int] = {}
    for split in splits:
        target_path = output_dir / f"catboost_{split}.pt"
        if target_path.exists() and not args.overwrite:
            raise FileExistsError(f"{target_path} exists. Re-run with --overwrite to replace it.")

        payloads = []
        for variant in variants:
            variant_path = source_dir / variant / f"catboost_{split}.pt"
            print(f"Loading {variant_path}")
            payloads.append(_torch_load(variant_path))

        print(f"Concatenating {len(payloads)} payloads for split='{split}'")
        stacked = _stack_payloads(payloads, seed=_stable_split_seed(args.seed, split))
        torch.save(stacked, target_path)
        split_sizes[split] = int(stacked["features"].shape[0])
        print(f"Saved master split -> {target_path}")

    base_metadata = _load_base_metadata(source_dir)
    _write_metadata(base_metadata, output_dir, variants, split_sizes, args.seed)
    print(f"Metadata written to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
