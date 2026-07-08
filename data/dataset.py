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

# Dynamically raise the file descriptor limit to avoid "Too many open files" errors with large shard counts
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65536
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)
    if target > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        logger.info(f"Raised open file limit (RLIMIT_NOFILE): {soft} → {target}")
except Exception as e:
    logger.debug(f"Failed to raise open file limit: {e}")


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
        
        # Index all files to get their shapes and sequence counts without keeping files open
        self.file_paths = []
        self.cumulative_lengths = [0]
        
        for f in npy_files:
            try:
                # Open briefly to inspect shape and calculate sequence count
                arr = np.load(str(f), mmap_mode='r')
                shape = arr.shape
                ndim = arr.ndim
                
                # Calculate n_seqs for this file
                if ndim == 1:
                    n_seqs = len(arr) // seq_len
                elif ndim == 2:
                    if shape[1] == seq_len:
                        n_seqs = shape[0]
                    else:
                        total_tokens = shape[0] * shape[1]
                        n_seqs = total_tokens // seq_len
                else:
                    logger.warning(f"File {f} has ndim={ndim}, expected 1 or 2. Skipping.")
                    continue
                
                if n_seqs == 0:
                    logger.warning(f"File {f}: too few tokens to form seq_len={seq_len}. Skipping.")
                    continue
                
                self.file_paths.append(str(f))
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + n_seqs)
                logger.debug(f"Indexed {f}: {n_seqs} sequences")
                
                # Let garbage collector close the file handle immediately
                del arr
            except Exception as e:
                logger.warning(f"Failed to index {f}: {e}")
        
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
        """Binary search for the correct file path and load slice dynamically."""
        lo, hi = 0, len(self.file_paths) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if global_idx < self.cumulative_lengths[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        
        local_idx = global_idx - self.cumulative_lengths[lo]
        file_path = self.file_paths[lo]
        
        # Open memmap file briefly to read the required segment
        arr = np.load(file_path, mmap_mode='r')
        
        token_start = local_idx * self.seq_len
        token_end = token_start + self.seq_len
        
        if arr.ndim == 1:
            tokens = np.array(arr[token_start:token_end])
        elif arr.ndim == 2:
            w_source = arr.shape[1]
            if w_source == self.seq_len:
                tokens = np.array(arr[local_idx])
            else:
                # Dynamic sub-slice mapping to prevent reading the entire file
                row_start = token_start // w_source
                row_end = (token_end - 1) // w_source
                sub_arr = arr[row_start : row_end + 1]
                flat = sub_arr.reshape(-1)
                
                col_start = token_start % w_source
                col_end = col_start + self.seq_len
                tokens = np.array(flat[col_start:col_end])
        else:
            raise ValueError(f"Unexpected array dimension {arr.ndim} in {file_path}")
            
        return tokens
    
    def __len__(self) -> int:
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx = int(self.indices[idx % len(self.indices)])
        tokens = self._get_item_from_mmap(global_idx).astype(np.int64)
        tokens_tensor = torch.from_numpy(tokens)
        
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