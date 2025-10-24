import os
import random
import sys

import numpy as np
import pandas as pd
from typing import Iterable
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.metadata import MetadataDataset
from dataset.work_operations import WorkOperationsDataset


class HackathonDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        download=False,
        sampling_strategy: list[dict] | None = None,
        root: str = "data",
        num_companies: int = 14,
        seed: int | None = None,
    ):
        np_rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)

        self.work_operations_dataset = WorkOperationsDataset(
            root=root,
            download=download,
            split=split,
            seed=seed,
            sampling_strategy=sampling_strategy
            or [
                {
                    "subset_size": 0.5,
                    "sample_pct": 0.5,
                    "use_balanced_data": True,
                    "use_sampled_calculus": True,
                },
                {
                    "subset_size": 0.5,
                    "sample_pct": 0.3,
                    "use_balanced_data": True,
                    "use_sampled_calculus": True,
                },
            ],
            np_rng=np_rng,
            py_rng=py_rng,
        )
        self.metadata_dataset = MetadataDataset(
            root=root,
            download=download,
            split=split,
            num_companies=num_companies,
        )

    def shuffle(self):
        self.work_operations_dataset.shuffle()

    def __len__(self):
        return len(self.work_operations_dataset)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        work_operations = self.work_operations_dataset[idx]  # type: ignore
        project = self.metadata_dataset[work_operations["project_id"]]
        return {**work_operations, **project}  # type: ignore

    def get_polars_dataframe(self):
        work_operations = self.work_operations_dataset.get_exploded_dataframe()
        projects = self.metadata_dataset.data
        return work_operations.join(projects, on="project_id", how="left")

    def get_pandas_dataframe(self):
        return self.get_polars_dataframe().to_pandas()

    def multi_hot_encode_list(self, label: list[int]) -> list[int]:
        return self.work_operations_dataset._index_encode(label)

    def multi_hot_encode_batch(self, labels: list[list[int]]) -> np.ndarray:
        return self.work_operations_dataset._index_encode_batch(labels)

    def to_cluster_names(self, it: Iterable) -> list[str]:
        return self.work_operations_dataset.to_cluster_names(it)

    def create_submission(self, predictions):
        """Create a submission file.
        Args:
            predictions: dict of id to list of predicted cluster ids
        """
        submission = [[k, *self.multi_hot_encode_list(v)] for k, v in predictions.items()]
        df = pd.DataFrame(
            submission,
            columns=["id"] + [str(i) for i in range(0, self.work_operations_dataset.num_clusters)],
        )
        assert df.shape[0] == 18299, "Submission must have 18299 rows."
        assert df.id.unique().shape[0] == 18299, "Submission IDs must be unique."
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"submissions/submission_{timestamp}.csv", index=False)
        print(f"Submission saved to submissions/submission_{timestamp}.csv")

    def __repr__(self):
        return f"HackathonDataset(split={self.work_operations_dataset.split}, length={len(self)})"

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    dataset = HackathonDataset(split="train", download=True, seed=42, root="data")
    test_predictions = {idx: [] for idx in range(18299)}
    dataset.create_submission(test_predictions)
