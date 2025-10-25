"""Utility script to cache HackathonDataset features/labels as NumPy archives."""

import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dataset.hackathon import HackathonDataset
    from dataset.collate import collate_fn
except ImportError:
    print("Error: Could not import HackathonDataset or collate_fn. Ensure 'dataset' is on PYTHONPATH.")
    raise


def _load_split(split: str, seed: int = 42):
    """Load an entire dataset split into memory and return NumPy arrays."""
    dataset = HackathonDataset(split=split, download=True, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))
    features = batch["X"].cpu().numpy()
    labels = batch["Y"].cpu().numpy()
    return features, labels


def main():
    print("Loading train split...")
    X_train, y_train = _load_split("train")
    print(f"Train tensors captured: X={X_train.shape}, y={y_train.shape}")

    print("Loading validation split...")
    X_val, y_val = _load_split("val")
    print(f"Val tensors captured: X={X_val.shape}, y={y_val.shape}")

    output_dir = Path(__file__).resolve().parent / "generated_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving compressed NumPy archives to {output_dir}...")
    np.savez_compressed(output_dir / "train_dataset.npz", X=X_train, y=y_train)
    np.savez_compressed(output_dir / "val_dataset.npz", X=X_val, y=y_val)
    print("Dataset snapshots saved.")


if __name__ == "__main__":
    main()
