"""
High-performance dataset classes for FORGE pretraining.
Uses memory-mapped numpy arrays for zero-copy data loading.
"""

from __future__ import annotations
import os
import math
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class PackedTokenDataset(Dataset):
    """
    Memory-mapped dataset of pre-tokenized, packed sequences.
    
    Expects .npy files containing arrays of shape (N, seq_len) with dtype uint32.
    Uses mmap for zero-copy access — no RAM required beyond OS page cache.
    """
    
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        split: str = "train",
        max_samples: Optional[int] = None,
        shuffle_files: bool = True,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.split = split
        
        data_path = Path(data_dir)
        npy_files = sorted(data_path.glob(f"**/{split}*.npy"))
        
        if not npy_files:
            npy_files = sorted(data_path.glob("**/*.npy"))
            logger.warning(f"No {split}-specific files found, using all .npy files")
        
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")
        
        # Load all files as memory-mapped arrays
        self.mmaps = []
        self.cumulative_lengths = [0]
        
        for f in npy_files:
            try:
                arr = np.load(str(f), mmap_mode='r')
                if arr.ndim == 1:
                    # Flat token array — reshape to sequences
                    n_seqs = len(arr) // seq_len
                    arr = arr[:n_seqs * seq_len].reshape(n_seqs, seq_len)
                
                if arr.shape[1] != seq_len:
                    logger.warning(f"File {f} has seq_len={arr.shape[1]}, expected {seq_len}. Skipping.")
                    continue
                
                self.mmaps.append(arr)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(arr))
                logger.debug(f"Loaded {f}: {len(arr)} sequences")
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        
        self.total_sequences = self.cumulative_lengths[-1]
        
        if max_samples:
            self.total_sequences = min(self.total_sequences, max_samples)
        
        # Shuffle index
        rng = np.random.default_rng(seed)
        self.indices = rng.permutation(self.total_sequences) if shuffle_files else \
                      np.arange(self.total_sequences)
        
        logger.info(f"PackedTokenDataset ({split}): "
                    f"{self.total_sequences:,} sequences × {seq_len} tokens = "
                    f"{self.total_sequences * seq_len / 1e9:.2f}B tokens total")
    
    def _get_item_from_mmap(self, global_idx: int) -> np.ndarray:
        """Binary search for the correct mmap file."""
        lo, hi = 0, len(self.mmaps) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if global_idx < self.cumulative_lengths[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        
        local_idx = global_idx - self.cumulative_lengths[lo]
        return self.mmaps[lo][local_idx]
    
    def __len__(self) -> int:
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx = int(self.indices[idx % len(self.indices)])
        tokens = self._get_item_from_mmap(global_idx).astype(np.int64)
        tokens_tensor = torch.from_numpy(tokens)  # zero-copy with numpy
        
        return {
            "input_ids": tokens_tensor,
            "labels": tokens_tensor.clone(),
        }


class WeightedDataMixer(Dataset):
    """
    Mix multiple datasets with configurable weights.
    Implements domain-weighted sampling for curriculum control.
    """
    
    def __init__(
        self,
        datasets: Dict[str, Dataset],
        weights: Dict[str, float],
        total_samples: Optional[int] = None,
        seed: int = 42,
    ):
        assert set(datasets.keys()) == set(weights.keys())
        
        self.datasets = datasets
        self.names = list(datasets.keys())
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.probs = [weights[n] / total_weight for n in self.names]
        
        # Determine total dataset size
        if total_samples is None:
            total_samples = sum(len(d) for d in datasets.values())
        self.total_samples = total_samples
        
        # Pre-generate sampling plan for reproducibility
        rng = np.random.default_rng(seed)
        self.plan = rng.choice(len(self.names), size=total_samples, p=self.probs)
        

        
        logger.info(f"WeightedDataMixer: {total_samples:,} total samples from "
                    f"{len(datasets)} sources with weights {weights}")
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds_name = self.names[self.plan[idx % len(self.plan)]]
        ds = self.datasets[ds_name]
        
        # Sample from dataset
        local_idx = idx % len(ds)
        return ds[local_idx]


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 8,
    prefetch_factor: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """
    Build a high-performance DataLoader with:
    - Pinned memory for fast GPU transfer
    - Multiple workers with prefetching
    - Persistent workers (avoid spawn overhead per epoch)
    """
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=True,           # avoid partial batches that break ZeRO-3 
        generator=generator,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
    )