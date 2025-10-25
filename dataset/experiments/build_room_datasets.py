import json
import sys
from pathlib import Path
from typing import Dict, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.work_operations import WorkOperationsDataset

DATA_ROOT = "data"
OUTPUT_DIR = PROJECT_ROOT / "precomputed" / "rooms"
DEFAULT_SEED = 42


def build_room_sequence(dataset: WorkOperationsDataset) -> tuple[list[str], dict[str, int]]:
    rooms = list(dataset.room_to_index.keys())
    if "ukjent" not in rooms:
        rooms.append("ukjent")
    return rooms, {room: idx for idx, room in enumerate(rooms)}


def aggregate_features(
    sample: dict,
    room_sequence: Sequence[str],
    room_to_index: dict[str, int],
    num_clusters: int,
) -> torch.Tensor | None:
    room_blocks = {
        room: torch.zeros(num_clusters, dtype=torch.float32)
        for room in room_sequence
    }

    target_room = sample["room_cluster"]
    if target_room not in room_blocks:
        return None
    room_blocks[target_room] = torch.maximum(
        room_blocks[target_room], sample["X"].to(torch.float32)
    )

    for ctx in sample["calculus"]:
        room = ctx["room_cluster"]
        if room not in room_blocks:
            continue
        room_blocks[room] = torch.maximum(
            room_blocks[room], ctx["work_operations_index_encoded"].to(torch.float32)
        )

    flattened = torch.cat([room_blocks[room] for room in room_sequence], dim=0)

    room_indicator = torch.zeros(len(room_sequence), dtype=torch.float32)
    room_indicator[room_to_index[target_room]] = 1.0

    return torch.cat([flattened, room_indicator], dim=0)


def collect_split(
    dataset: WorkOperationsDataset,
    room_sequence: list[str],
    room_to_index: dict[str, int],
) -> Dict[str, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    ids: list[int] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        feat = aggregate_features(sample, room_sequence, room_to_index, dataset.num_clusters)
        if feat is None:
            continue
        features.append(feat)
        labels.append(sample["Y"].to(torch.float32))
        ids.append(int(sample["id"]))

    return {
        "features": torch.stack(features),
        "labels": torch.stack(labels),
        "ids": torch.tensor(ids, dtype=torch.int64),
    }


def save_split(data: Dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def main() -> None:
    train_base = WorkOperationsDataset(
        root=DATA_ROOT,
        split="train",
        download=False,
        seed=DEFAULT_SEED,
    )
    room_sequence, room_to_index = build_room_sequence(train_base)

    splits = {
        "train": train_base,
        "val": WorkOperationsDataset(
            root=DATA_ROOT,
            split="val",
            download=False,
            seed=DEFAULT_SEED,
        ),
        "test": WorkOperationsDataset(
            root=DATA_ROOT,
            split="test",
            download=False,
            seed=DEFAULT_SEED,
        ),
    }

    metadata = {
        "room_sequence": room_sequence,
        "num_clusters": train_base.num_clusters,
        "feature_dim": len(room_sequence) * train_base.num_clusters + len(room_sequence),
        "room_indicator_dim": len(room_sequence),
        "seed": DEFAULT_SEED,
        "label_type": "hidden_only",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2, ensure_ascii=False)

    for split_name, dataset in splits.items():
        split_data = collect_split(dataset, room_sequence, room_to_index)
        save_path = OUTPUT_DIR / f"rooms_{split_name}.pt"
        save_split(split_data, save_path)
        print(
            f"Saved {split_name} split with {split_data['features'].shape[0]} samples -> {save_path}"
        )


if __name__ == "__main__":
    main()
