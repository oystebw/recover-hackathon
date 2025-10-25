# export_test_npz.py
"""
Export the Hackathon test split to a single NPZ:
  - X: 2D float32 array of features (concatenates 'X' and 'room_cluster_one_hot' → 399 cols)
  - ids: 1D int64 array of submission row ids
  - num_submission_rows: int (for building the final CSV on Kaggle)

Usage:
  python export_test_npz.py --out data/test_dataset.npz --num-rows 18299
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# ---- Import your dataset module ----
try:
    from dataset.hackathon import HackathonDataset
except Exception as e:
    raise SystemExit(
        "Could not import dataset.hackathon.HackathonDataset. "
        "Run from repo root or add the project to PYTHONPATH.\n"
        f"Underlying error: {e}"
    )

def predict_collate_fn(batch):
    """Build the 399-dim feature vector per item: concat(item['X'], item['room_cluster_one_hot'])."""
    ids, feats = [], []
    for item in batch:
        try:
            x = item["X"]                         # shape [388]
            rc = item["room_cluster_one_hot"]     # shape [11]
            feats.append(torch.cat((x, rc), dim=0))
            ids.append(int(item["id"]))
        except KeyError as ke:
            raise KeyError(f"Missing key {ke} in dataset item with id={item.get('id', '?')}")
    return {"id": torch.tensor(ids, dtype=torch.long),
            "X": torch.stack(feats, dim=0).to(torch.float32)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/test_dataset.npz"),
                    help="Output NPZ path")
    ap.add_argument("--num-rows", type=int, default=18299,
                    help="Total submission rows (e.g., max id + 1)")
    args = ap.parse_args()

    # Load full test split
    ds = HackathonDataset(split="test", download=True, seed=42)
    if len(ds) == 0:
        raise SystemExit("Test split is empty.")

    dl = DataLoader(ds, batch_size=len(ds), collate_fn=predict_collate_fn)
    batch = next(iter(dl))
    X = batch["X"].numpy().astype(np.float32)         # [N, 399]
    ids = batch["id"].numpy().astype(np.int64)        # [N]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, ids=ids, num_submission_rows=np.array(args.num_rows, dtype=np.int64))

    print(f"Saved: {args.out}")
    print(f"X shape: {X.shape}  ids: {ids.shape}  num_submission_rows: {args.num_rows}")
    print("Upload this NPZ to Kaggle and point your notebook/script to it.")

if __name__ == "__main__":
    main()