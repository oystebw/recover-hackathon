import os
import typing
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    _repr_indent = 4

    def __init__(self, root: str | Path):
        if isinstance(root, str):
            root = os.path.expanduser(root)
        self.root = root

    def __getitem__(self, index: int) -> typing.Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Data location: {self.root}")
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def _format_transform_repr(
        self, transform: typing.Callable, head: str
    ) -> list[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + [
            "{}{}".format(" " * len(head), line) for line in lines[1:]
        ]

    def extra_repr(self) -> str:
        return ""


def index_encode_str(label: str, length: int, lookup: dict[str, int]) -> list[int]:
    vec = np.zeros(length, dtype=np.int8)
    if label in lookup.keys():
        vec[lookup[label]] = 1
    return vec.tolist()


def index_encode_str_batch(
    labels: list[str], length: int, lookup: dict[str, int]
) -> np.ndarray:
    vec = np.zeros((len(labels), length), dtype=np.int8)
    for i, label in enumerate(labels):
        if label in lookup.keys():
            vec[i, lookup[label]] = 1
    return vec
