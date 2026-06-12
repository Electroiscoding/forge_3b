"""
FORGE-3B Data Collators.

Provides batching utilities that bridge the tokenizer output format with what
the model and training engines expect. Three collators are provided:

  PretrainCollator  — packs uint32 pre-tokenized sequences; no padding needed
  SFTCollator       — pads to max length in batch; masks prompt tokens (-100)
  DPOCollator       — handles (chosen, rejected) pairs with independent padding

All collators return BF16-compatible long-int tensors on CPU; the training
engine moves them to the correct device with non_blocking=True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Pretrain Collator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PretrainCollator:
    """
    Collator for pre-packed, fixed-length sequences from PackedTokenDataset.

    Input samples are dicts with 'input_ids' and 'labels' (torch.long tensors
    of identical length seq_len). This collator simply stacks them — no padding
    required because all sequences are the same length by construction.
    """

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([s["input_ids"] for s in samples], dim=0)  # (B, T)
        labels    = torch.stack([s["labels"]    for s in samples], dim=0)  # (B, T)
        return {"input_ids": input_ids, "labels": labels}


# ──────────────────────────────────────────────────────────────────────────────
# SFT Collator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SFTCollator:
    """
    Collator for Supervised Fine-Tuning.

    Pads variable-length sequences within a batch to the length of the longest
    sequence. Prompt tokens are already masked to -100 by the dataset; this
    collator propagates that masking for padded positions.

    Args:
        pad_token_id: token ID used for input padding (model ignores with attn mask)
        max_length:   hard cap — sequences longer than this are truncated
        label_pad_id: value used to pad labels (default -100 = ignored by cross-entropy)
    """

    pad_token_id: int = 0
    max_length:   int = 4096
    label_pad_id: int = -100

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [s["input_ids"][: self.max_length] for s in samples]
        labels_list    = [s["labels"][: self.max_length]    for s in samples]

        max_len = max(t.size(0) for t in input_ids_list)

        padded_input  = torch.full((len(samples), max_len), self.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((len(samples), max_len), self.label_pad_id, dtype=torch.long)
        attn_mask     = torch.zeros((len(samples), max_len), dtype=torch.long)

        for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = ids.size(0)
            padded_input[i, :seq_len]  = ids
            padded_labels[i, :seq_len] = labs
            attn_mask[i, :seq_len]     = 1

        return {
            "input_ids":      padded_input,
            "labels":         padded_labels,
            "attention_mask": attn_mask,
        }


# ──────────────────────────────────────────────────────────────────────────────
# DPO Collator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DPOCollator:
    """
    Collator for Direct Preference Optimization.

    Each sample contains four tensors:
      chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    We pad chosen and rejected independently to their respective batch maximums
    so GPU memory is not wasted when chosen and rejected have very different lengths.

    Args:
        pad_token_id: used to pad input_ids
        max_length:   hard truncation cap per sequence
    """

    pad_token_id: int = 0
    max_length:   int = 4096
    label_pad_id: int = -100

    def _pad_batch(
        self,
        ids_list:    List[torch.Tensor],
        labels_list: List[torch.Tensor],
    ):
        max_len = max(t.size(0) for t in ids_list)
        B = len(ids_list)

        padded_ids    = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((B, max_len), self.label_pad_id, dtype=torch.long)
        attn_mask     = torch.zeros((B, max_len), dtype=torch.long)

        for i, (ids, labs) in enumerate(zip(ids_list, labels_list)):
            L = ids.size(0)
            padded_ids[i, :L]    = ids
            padded_labels[i, :L] = labs
            attn_mask[i, :L]     = 1

        return padded_ids, padded_labels, attn_mask

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        chosen_ids   = [s["chosen_input_ids"][: self.max_length]   for s in samples]
        chosen_labs  = [s["chosen_labels"][: self.max_length]      for s in samples]
        rej_ids      = [s["rejected_input_ids"][: self.max_length]  for s in samples]
        rej_labs     = [s["rejected_labels"][: self.max_length]     for s in samples]

        c_ids, c_labels, c_mask = self._pad_batch(chosen_ids, chosen_labs)
        r_ids, r_labels, r_mask = self._pad_batch(rej_ids,    rej_labs)

        return {
            "chosen_input_ids":        c_ids,
            "chosen_labels":           c_labels,
            "chosen_attention_mask":   c_mask,
            "rejected_input_ids":      r_ids,
            "rejected_labels":         r_labels,
            "rejected_attention_mask": r_mask,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_collator(
    mode: str,
    pad_token_id: int = 0,
    max_length: int = 4096,
):
    """
    Factory returning the correct collator for a given training mode.

    Args:
        mode:         'pretrain' | 'sft' | 'dpo'
        pad_token_id: token ID used for padding inputs
        max_length:   maximum sequence length (hard truncation)
    """
    if mode == "pretrain":
        return PretrainCollator()
    elif mode == "sft":
        return SFTCollator(pad_token_id=pad_token_id, max_length=max_length)
    elif mode == "dpo":
        return DPOCollator(pad_token_id=pad_token_id, max_length=max_length)
    else:
        raise ValueError(
            f"Unknown collator mode: '{mode}'. Expected 'pretrain', 'sft', or 'dpo'."
        )
