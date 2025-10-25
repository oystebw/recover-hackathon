import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PRECOMPUTED_DIR = PROJECT_ROOT / "precomputed" / "rooms"
SPLITS = ("train", "val", "test")


class RoomDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        ids: torch.Tensor | None = None,
    ) -> None:
        self.features = features
        self.labels = labels
        default_ids = torch.full((features.shape[0],), -1, dtype=torch.int64)
        self.ids = ids if ids is not None else default_ids

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class RoomPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MetricRoomLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        metric_weight: float = 1.0,
        fp_weight: float = 0.25,
        fn_weight: float = 0.5,
        empty_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.metric_weight = metric_weight
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.empty_penalty = empty_penalty

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.bce_weight:
            loss = loss + self.bce_weight * self.bce(logits, targets)

        if self.metric_weight:
            probs = torch.sigmoid(logits)
            tp = (targets * probs).sum(dim=1)
            fp = ((1 - targets) * probs).sum(dim=1)
            fn = (targets * (1 - probs)).sum(dim=1)

            room_metric = tp - self.fp_weight * fp - self.fn_weight * fn

            if self.empty_penalty:
                empty_mask = (targets.sum(dim=1) == 0).float()
                empties = empty_mask * probs.sum(dim=1)
                room_metric = room_metric - self.empty_penalty * empties

            metric_loss = -room_metric.mean()
            loss = loss + self.metric_weight * metric_loss

        return loss


@dataclass
class TrainingArtifacts:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    model: RoomPredictor
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    metadata: dict


def _load_metadata(precomputed_dir: Path) -> dict:
    metadata_path = precomputed_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}.")
    with metadata_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_split(precomputed_dir: Path, split: str) -> RoomDataset:
    split_path = precomputed_dir / f"rooms_{split}.pt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split '{split}' not found at {split_path}.")

    payload = torch.load(split_path)
    return RoomDataset(
        payload["features"].to(torch.float32),
        payload["labels"].to(torch.float32),
        payload.get("ids"),
    )


def load_precomputed_datasets(
    precomputed_dir: Path | str = DEFAULT_PRECOMPUTED_DIR,
) -> tuple[RoomDataset, RoomDataset, RoomDataset, dict]:
    precomputed_dir = Path(precomputed_dir)
    metadata = _load_metadata(precomputed_dir)
    datasets = [_load_split(precomputed_dir, split) for split in SPLITS]
    return (*datasets, metadata)


def prepare_training(
    precomputed_dir: Path | str = DEFAULT_PRECOMPUTED_DIR,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> TrainingArtifacts:
    train_dataset, val_dataset, test_dataset, metadata = load_precomputed_datasets(
        precomputed_dir
    )

    input_dim = metadata["feature_dim"]
    output_dim = metadata["num_clusters"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RoomPredictor(input_dim=input_dim, output_dim=output_dim).to(device)
    criterion = MetricRoomLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    return TrainingArtifacts(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        metadata=metadata,
    )


def train_one_epoch(artifacts: TrainingArtifacts, epoch: int, log_every: int = 50) -> float:
    model = artifacts.model
    criterion = artifacts.criterion
    optimizer = artifacts.optimizer
    device = artifacts.device

    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch_idx, (features, targets) in enumerate(artifacts.train_loader, start=1):
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

        if batch_idx % log_every == 0 or batch_idx == len(artifacts.train_loader):
            avg_loss = total_loss / total_batches
            print(
                f"  Epoch {epoch:02d} | Batch {batch_idx:4d}/{len(artifacts.train_loader):4d} | "
                f"loss={loss.item():.4f} | running_avg={avg_loss:.4f}"
            )

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(artifacts: TrainingArtifacts, loader: DataLoader) -> float:
    model = artifacts.model
    criterion = artifacts.criterion
    device = artifacts.device

    model.eval()
    total_loss = 0.0
    total_batches = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        loss = criterion(logits, targets)

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


def main() -> None:
    precomputed_dir = Path(os.environ.get("ROOM_DATA_DIR", DEFAULT_PRECOMPUTED_DIR))
    artifacts = prepare_training(precomputed_dir=precomputed_dir)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(artifacts, epoch=epoch)
        val_loss = evaluate(artifacts, artifacts.val_loader)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    test_loss = evaluate(artifacts, artifacts.test_loader)
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
