import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.work_operations import WorkOperationsDataset


DATA_DIR = PROJECT_ROOT / "precomputed" / "rooms"
SPLITS = ["rooms_train", "rooms_val", "rooms_test"]


def summarize_split(split_name: str, code_to_name: dict[int, str], room_indicator_dim: int) -> None:
    path = DATA_DIR / f"{split_name}.pt"
    if not path.exists():
        print(f"Split '{split_name}' is missing at {path} – run kjokken_prepare_data.py first.")
        return

    payload = torch.load(path)
    features = payload["features"]
    labels = payload["labels"]
    ids = payload["ids"]

    print(f"\nSplit: {split_name}")
    print(f"  features: {tuple(features.shape)} (float)")
    print(f"  labels:   {tuple(labels.shape)} (float)")
    print(f"  ids:      {tuple(ids.shape)} (int64)")

    example_idx = 0
    feat_sample = features[example_idx]
    label_sample = labels[example_idx]
    sample_id = int(ids[example_idx])

    num_clusters = label_sample.shape[0]
    flattened_dim = feat_sample.shape[0] - room_indicator_dim
    num_rooms = flattened_dim // num_clusters if room_indicator_dim else flattened_dim // num_clusters

    print(f"  Example id: {sample_id}")
    print(f"    Flattened rooms: {num_rooms}")
    print(f"    Clusters per room: {num_clusters}")

    # Inspect positive labels (complete kitchen operation set)
    hidden_ops = torch.nonzero(label_sample, as_tuple=False).flatten().tolist()
    op_names = [code_to_name.get(code, f"Unknown {code}") for code in hidden_ops]
    print(f"    Hidden operations ({len(hidden_ops)}): {op_names}")


def main() -> None:
    if not DATA_DIR.exists():
        print(f"Data directory {DATA_DIR} not found. Run kjokken_prepare_data.py first.")
        return

    room_indicator_dim = 0
    metadata_path = DATA_DIR / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fp:
            metadata = json.load(fp)
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        room_indicator_dim = metadata.get("room_indicator_dim", 0)
    else:
        print("metadata.json missing – proceeding without metadata summary.")

    # Load mapping for operation names
    wo_dataset = WorkOperationsDataset(root="data", split="train", download=False)
    code_to_name = wo_dataset.code_to_wo

    for split in SPLITS:
        summarize_split(split, code_to_name, room_indicator_dim)


if __name__ == "__main__":
    main()
