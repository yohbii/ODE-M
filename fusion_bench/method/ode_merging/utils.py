from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPProcessor
from fusion_bench.dataset import CLIPDataset




# @dataclass(frozen=True)
# class CalibrationSpec:
#     batch_size: int = 32
#     num_samples: Optional[int] = 1024
#     num_batches: Optional[int] = None
#     shuffle: bool = True
#     drop_last: bool = False
#     num_workers: int = 0
#     pin_memory: bool = False
#     persistent_workers: bool = False
#     collate_fn: Optional[Callable[[Sequence[Any]], Any]] = None


def make_calibration_dataloader(
    dataset: Union[Dataset, Any],
    processor: CLIPProcessor,
    *,
    batch_size: int,
    num_samples: Optional[int] = 1024,
    num_batches: Optional[int] = None,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[Sequence[Any]], Any]] = None,
) -> DataLoader:
    """
    Create a calibration DataLoader by sampling a subset from the input dataset.

    Args:
        dataset: A PyTorch Dataset (implements __len__ and __getitem__).
                 If you pass a HuggingFace datasets.Dataset, it also works if it supports __len__/__getitem__.
        batch_size: Batch size of the returned calibration loader.
        num_samples: Number of examples to sample for calibration (default 1024).
        num_batches: If provided, overrides num_samples with num_batches * batch_size.
        shuffle: Whether to sample randomly (default True). If False, take the first N samples.
        seed: Random seed for sampling.
        drop_last: DataLoader drop_last.
        num_workers: DataLoader num_workers.
        pin_memory: DataLoader pin_memory.
        persistent_workers: DataLoader persistent_workers (effective if num_workers > 0).
        collate_fn: Optional DataLoader collate_fn.

    Returns:
        A DataLoader over a subset of the original dataset.

    Notes:
        - This function samples WITHOUT replacement.
        - If requested samples exceed dataset length, it will cap at len(dataset).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    if num_batches is not None:
        if num_batches <= 0:
            raise ValueError(f"num_batches must be > 0, got {num_batches}")
        target_n = num_batches * batch_size
    else:
        if num_samples is None or num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        target_n = num_samples

    # Length check
    try:
        n_total = len(dataset)
    except TypeError as e:
        raise TypeError("dataset must implement __len__ for sampling a subset.") from e

    if n_total <= 0:
        raise ValueError("dataset appears to be empty (len(dataset) <= 0).")

    n = min(target_n, n_total)

    if shuffle:
        indices = torch.randperm(n_total)[:n].tolist()
    else:
        indices = list(range(n))

    subset = Subset(dataset, indices)
    subset = CLIPDataset(subset, processor)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        collate_fn=collate_fn,
    )
    return loader


def get_param_names_to_merge(
    input_param_names: List[str], exclude_param_names_regex: list
) -> List[str]:
    """
    Get the names of parameters that need to be merged.

    Args:
        input_param_names (List[str]): List of input parameter names.
        exclude_param_names_regex (list): List of regular expressions for parameter names to be excluded.

    Returns:
        List[str]: List of parameter names to be merged.
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any(
            [
                re.match(exclude_pattern, param_name)
                for exclude_pattern in exclude_param_names_regex
            ]
        )
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_param(
        model: nn.Module, param_names_to_merge: List[str]
) -> Dict[str, Tensor]:
    """
    Get the references of parameters.

    Args:
        model (nn.Module): The model.
        param_names_to_merge (List[str]): List of parameter names to be merged.

    Returns:
        Dict[str, Tensor]: Dictionary of parameter names and their references.
    """
    param_dict = {
        param_name: param_value
        for param_name, param_value in model.state_dict(keep_vars=True).items()
        if param_name in param_names_to_merge
    }
    return param_dict

def get_param_gradients(
    model: nn.Module, param_names_to_merge: List[str]
) -> Dict[str, Tensor]:
    """
    Get the gradients of parameters.

    Args:
        model (nn.Module): The model.
        param_names_to_merge (List[str]): List of parameter names to be merged.

    Returns:
        Dict[str, Tensor]: Dictionary of parameter names and their squared gradients.
    """
    param_gradients = {
        param_name: param_value.grad.detach()
        for param_name, param_value in model.state_dict(keep_vars=True).items()
        if param_name in param_names_to_merge
    }
    return param_gradients

class TaskTaggedDataset(Dataset):
    def __init__(self, ds, task_name: str):
        self.ds = ds
        self.task_name = task_name

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = self.ds[idx]
        image, label = x
        return image, (label, self.task_name)

def sample_subset(ds, n):
    n = min(n, len(ds))
    idx = torch.randperm(len(ds))[:n].tolist()
    return Subset(ds, idx)