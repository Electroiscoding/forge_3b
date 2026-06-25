"""
CRAYON Tokenizer Wrapper
Wraps CrayonVocab into a full-featured tokenizer compatible with FORGE training.
Exploits CRAYON's 24M+ tok/sec CPU throughput via parallel multi-process pipeline.
"""

from __future__ import annotations

import os
import gc
import time
import logging
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Union, Optional, Dict, Any, Iterator, Tuple
from pathlib import Path
import queue

# Fix Windows console UTF-8 printing issues for external packages (e.g., Crayon print statements with emoji)
import sys
import io
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import numpy as np

from crayon import CrayonVocab

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# THREAD-LOCAL TOKENIZER POOL
# Each worker thread gets its own CrayonVocab instance to avoid any GIL
# contention on the underlying C++ DAT engine.
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()
_init_lock = threading.Lock()

def _get_thread_local_tokenizer(profile: str = "standard") -> CrayonVocab:
    """Return a thread-local CrayonVocab instance, creating one if needed."""
    if not hasattr(_thread_local, "tokenizer"):
        with _init_lock:
            # CPU is 20× faster than CUDA for CRAYON (23.84M vs 1.22M tok/s)
            _thread_local.tokenizer = CrayonVocab(device="cpu")
            _thread_local.tokenizer.load_profile(profile)
            logger.debug(f"Thread {threading.current_thread().name}: "
                         f"CrayonVocab loaded with profile='{profile}'")
    return _thread_local.tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# SPECIAL TOKEN REGISTRY
# We manage special tokens separately since CRAYON's vocabulary may not have
# them natively — we append them to the vocab ID space.
# ─────────────────────────────────────────────────────────────────────────────

SPECIAL_TOKENS = {
    "<|pad|>":       0,
    "<|bos|>":       1,
    "<|eos|>":       2,
    "<|sep|>":       3,
    "<|sys|>":       4,
    "<|/sys|>":      5,
    "<|usr|>":       6,
    "<|/usr|>":      7,
    "<|asst|>":      8,
    "<|/asst|>":     9,
    "<|tool|>":      10,
    "<|tool_resp|>": 11,
    "<|unk|>":       12,
}

# For the standard profile: actual vocab starts at ID 13
# Effective vocab_size = 206,373 (crayon) + 13 (specials) = 206,386 → pad to 206,464
SPECIAL_TOKEN_OFFSET = len(SPECIAL_TOKENS)   # 13
CRAYON_STANDARD_VOCAB = 206_373
CRAYON_LITE_VOCAB = 50_000

# Reverse mapping for decode
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}


class ForgeTokenizer:
    """
    Production-grade FORGE tokenizer built on top of CRAYON.
    
    Features:
    - Thread-safe, zero-copy multi-threaded tokenization
    - Batched encoding with automatic chunking for throughput
    - Full special token support with chat template
    - Streaming decode
    - Vocabulary size adapter between CRAYON profile and model vocab
    - Offline preprocessing at 24M+ tokens/sec on CPU
    
    Usage:
        tokenizer = ForgeTokenizer(profile="standard")
        ids = tokenizer.encode("Hello world")
        text = tokenizer.decode(ids)
        
        # Chat format
        ids = tokenizer.encode_chat([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ])
    """
    
    def __init__(
        self,
        profile: str = "standard",
        device: str = "cpu",
        n_workers: int = None,
        max_length: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.profile = profile
        self.device_str = device   # "cpu" recommended for CRAYON throughput
        self.n_workers = n_workers or max(1, mp.cpu_count() - 2)
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Primary tokenizer (used for single-threaded ops)
        self._tokenizer = CrayonVocab(device=device)
        self._tokenizer.load_profile(profile)
        
        # Special tokens
        self.special_tokens = SPECIAL_TOKENS.copy()
        self.id_to_special = ID_TO_SPECIAL.copy()
        self.special_token_offset = SPECIAL_TOKEN_OFFSET
        
        # Derive vocab size
        if profile == "standard":
            self._crayon_vocab_size = CRAYON_STANDARD_VOCAB
        else:
            self._crayon_vocab_size = CRAYON_LITE_VOCAB
        
        # Effective vocab size = specials + crayon tokens
        # Padded to nearest multiple of 128 for efficient matmuls
        raw_vocab = self._crayon_vocab_size + SPECIAL_TOKEN_OFFSET
        self.vocab_size = math.ceil(raw_vocab / 128) * 128
        
        # Lock for thread safety on single-instance operations
        self._lock = threading.Lock()
        
        # Thread pool for parallel tokenization
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.n_workers,
            thread_name_prefix="forge_tok"
        )
        
        logger.info(
            f"ForgeTokenizer initialized: profile={profile}, "
            f"vocab_size={self.vocab_size} (crayon={self._crayon_vocab_size} "
            f"+ specials={SPECIAL_TOKEN_OFFSET} + padding={self.vocab_size - raw_vocab}), "
            f"n_workers={self.n_workers}, throughput ~24M tok/s (CPU)"
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # TOKEN ID CONVERSION HELPERS
    # CRAYON returns IDs in [0, crayon_vocab_size).
    # We shift them by SPECIAL_TOKEN_OFFSET to [SPECIAL_TOKEN_OFFSET, vocab_size).
    # Special tokens occupy IDs [0, SPECIAL_TOKEN_OFFSET).
    # ─────────────────────────────────────────────────────────────────────────
    
    def _shift_crayon_ids(self, ids: List[int]) -> List[int]:
        """Shift CRAYON token IDs to make room for special tokens at the front."""
        return [i + self.special_token_offset for i in ids]
    
    def _unshift_crayon_ids(self, ids: List[int]) -> List[int]:
        """Inverse of _shift_crayon_ids for decode path."""
        out = []
        for i in ids:
            if i < self.special_token_offset:
                out.append(None)   # special token — handled separately
            else:
                out.append(i - self.special_token_offset)
        return out
    
    # ─────────────────────────────────────────────────────────────────────────
    # CORE ENCODE / DECODE
    # ─────────────────────────────────────────────────────────────────────────
    
    def encode(
        self,
        text: str,
        add_bos: Optional[bool] = None,
        add_eos: Optional[bool] = None,
        truncate: bool = True,
    ) -> List[int]:
        """
        Encode a single string to token IDs.
        Thread-safe via CRAYON's internal lock or thread-local tokenizers.
        """
        add_bos_ = self.add_bos if add_bos is None else add_bos
        add_eos_ = self.add_eos if add_eos is None else add_eos
        
        # CRAYON tokenization
        with self._lock:
            raw_ids = self._tokenizer.tokenize(text)
        
        # Handle both list and tensor returns from CRAYON
        if isinstance(raw_ids, torch.Tensor):
            raw_ids = raw_ids.tolist()
        
        ids = self._shift_crayon_ids(raw_ids)
        
        # Prepend/append special tokens
        if add_bos_:
            ids = [self.special_tokens["<|bos|>"]] + ids
        if add_eos_:
            ids = ids + [self.special_tokens["<|eos|>"]]
        
        # Truncate if needed
        if truncate and self.max_length is not None:
            ids = ids[:self.max_length]
        
        return ids
    
    def encode_batch(
        self,
        texts: List[str],
        add_bos: Optional[bool] = None,
        add_eos: Optional[bool] = None,
        pad: bool = True,
        pad_to_multiple_of: int = 8,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of texts in parallel using thread pool.
        Leverages CRAYON's 24M tok/s CPU throughput across all threads.
        
        Returns dict with 'input_ids' and 'attention_mask'.
        """
        add_bos_ = self.add_bos if add_bos is None else add_bos
        add_eos_ = self.add_eos if add_eos is None else add_eos
        
        # ── Parallel tokenization ──────────────────────────────────────────
        def _tokenize_one(text: str) -> List[int]:
            tok = _get_thread_local_tokenizer(self.profile)
            raw = tok.tokenize(text)
            if isinstance(raw, torch.Tensor):
                raw = raw.tolist()
            ids = self._shift_crayon_ids(raw)
            if add_bos_:
                ids = [self.special_tokens["<|bos|>"]] + ids
            if add_eos_:
                ids = ids + [self.special_tokens["<|eos|>"]]
            if self.max_length is not None:
                ids = ids[:self.max_length]
            return ids
        
        if self.n_workers <= 1:
            # Run sequentially on the main thread using the primary tokenizer instance
            # to completely avoid any C++ multi-threading deadlocks.
            all_ids = []
            for t in texts:
                raw = self._tokenizer.tokenize(t)
                if isinstance(raw, torch.Tensor):
                    raw = raw.tolist()
                ids = self._shift_crayon_ids(raw)
                if add_bos_:
                    ids = [self.special_tokens["<|bos|>"]] + ids
                if add_eos_:
                    ids = ids + [self.special_tokens["<|eos|>"]]
                if self.max_length is not None:
                    ids = ids[:self.max_length]
                all_ids.append(ids)
        else:
            # Submit all texts to thread pool
            futures = [self._thread_pool.submit(_tokenize_one, t) for t in texts]
            all_ids = [f.result() for f in futures]
        
        if not pad or return_tensors is None:
            return {"input_ids": all_ids}
        
        # ── Padding ────────────────────────────────────────────────────────
        max_len = max(len(ids) for ids in all_ids)
        # Pad to multiple for efficiency
        if pad_to_multiple_of > 1:
            max_len = math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of
        
        pad_id = self.special_tokens["<|pad|>"]
        padded_ids = []
        attention_masks = []
        
        for ids in all_ids:
            n_pad = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * n_pad)
            attention_masks.append([1] * len(ids) + [0] * n_pad)
        
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        return {"input_ids": padded_ids, "attention_mask": attention_masks}
    
    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        # Separate special tokens from CRAYON tokens
        crayon_ids = []
        special_strs = []
        
        for tok_id in ids:
            if tok_id < self.special_token_offset:
                if not skip_special_tokens and tok_id in self.id_to_special:
                    special_strs.append(self.id_to_special[tok_id])
            else:
                crayon_ids.append(tok_id - self.special_token_offset)
        
        # CRAYON decode
        if crayon_ids:
            with self._lock:
                text = self._tokenizer.decode(crayon_ids)
        else:
            text = ""
        
        # Re-insert special tokens if not skipping
        if special_strs and not skip_special_tokens:
            text = " ".join(special_strs) + " " + text
        
        return text
    
    def decode_batch(
        self,
        batch_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch in parallel."""
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.tolist()
        
        if self.n_workers <= 1:
            return [self.decode(ids, skip_special_tokens) for ids in batch_ids]
            
        futures = [
            self._thread_pool.submit(self.decode, ids, skip_special_tokens)
            for ids in batch_ids
        ]
        return [f.result() for f in futures]
    
    # ─────────────────────────────────────────────────────────────────────────
    # CHAT TEMPLATE
    # ─────────────────────────────────────────────────────────────────────────
    
    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        return_as_tensor: bool = True,
    ) -> Union[List[int], torch.Tensor]:
        """
        Encode a conversation using FORGE chat template.
        
        Template:
            <|bos|><|sys|>{system}<|/sys|>
            <|usr|>{user}<|/usr|>
            <|asst|>{assistant}<|/asst|>
            ...
        
        Args:
            messages: List of {"role": "system|user|assistant", "content": str}
            add_generation_prompt: If True, append <|asst|> at end (inference)
        """
        token_ids = [self.special_tokens["<|bos|>"]]
        
        role_to_open = {
            "system":    "<|sys|>",
            "user":      "<|usr|>",
            "assistant": "<|asst|>",
        }
        role_to_close = {
            "system":    "<|/sys|>",
            "user":      "<|/usr|>",
            "assistant": "<|/asst|>",
        }
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Open tag
            open_tag = role_to_open.get(role, "<|usr|>")
            close_tag = role_to_close.get(role, "<|/usr|>")
            
            token_ids.append(self.special_tokens.get(open_tag, self.special_tokens["<|unk|>"]))
            
            # Encode content (no bos/eos, no truncation)
            content_ids = self.encode(content, add_bos=False, add_eos=False, truncate=False)
            token_ids.extend(content_ids)
            
            token_ids.append(self.special_tokens.get(close_tag, self.special_tokens["<|unk|>"]))
        
        if add_generation_prompt:
            token_ids.append(self.special_tokens["<|asst|>"])
        
        if self.max_length is not None:
            token_ids = token_ids[:self.max_length]
        
        if return_as_tensor:
            return torch.tensor(token_ids, dtype=torch.long)
        return token_ids
    
    def encode_chat_with_labels(
        self,
        messages: List[Dict[str, str]],
        loss_on_prompt: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a conversation and produce labels for SFT.
        Labels are -100 (ignored) for non-assistant turns unless loss_on_prompt=True.
        
        Returns dict with 'input_ids' and 'labels'.
        """
        input_ids = []
        labels = []
        
        input_ids.append(self.special_tokens["<|bos|>"])
        labels.append(-100 if not loss_on_prompt else self.special_tokens["<|bos|>"])
        
        role_to_open = {
            "system":    "<|sys|>",
            "user":      "<|usr|>",
            "assistant": "<|asst|>",
        }
        role_to_close = {
            "system":    "<|/sys|>",
            "user":      "<|/usr|>",
            "assistant": "<|/asst|>",
        }
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            is_assistant = (role == "assistant")
            
            open_tag_id = self.special_tokens.get(role_to_open.get(role, "<|usr|>"), 
                                                    self.special_tokens["<|unk|>"])
            close_tag_id = self.special_tokens.get(role_to_close.get(role, "<|/usr|>"),
                                                     self.special_tokens["<|unk|>"])
            
            content_ids = self.encode(content, add_bos=False, add_eos=False, truncate=False)
            
            turn_ids = [open_tag_id] + content_ids + [close_tag_id]
            input_ids.extend(turn_ids)
            
            if is_assistant or loss_on_prompt:
                # Compute loss on assistant content + close tag, NOT on open tag
                turn_labels = [-100] + content_ids + [close_tag_id]
                labels.extend(turn_labels)
            else:
                labels.extend([-100] * len(turn_ids))
        
        if self.max_length is not None:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # HIGH-THROUGHPUT OFFLINE PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    
    def tokenize_file_streaming(
        self,
        input_path: str,
        output_path: str,
        chunk_size: int = 100_000,       # documents per chunk
        seq_len: int = 2048,
        append_eos: bool = True,
        n_processes: int = None,
    ) -> int:
        """
        High-throughput offline tokenization of a large text file.
        
        Architecture:
        - Producer process reads text lines
        - N worker processes each run their own CrayonVocab instance
        - Consumer aggregates and packs sequences
        
        Throughput: ~5-10GB/min depending on hardware.
        Returns: total number of tokens written.
        """
        n_processes = n_processes or max(1, mp.cpu_count() - 2)
        total_tokens = 0
        
        logger.info(f"Offline tokenization: {input_path} → {output_path}, "
                    f"n_processes={n_processes}, seq_len={seq_len}")
        
        # We use process-based parallelism to bypass GIL fully
        with ProcessPoolExecutor(max_workers=n_processes) as pool:
            with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
                chunk_buffer = []
                futures = []
                
                for line in fin:
                    chunk_buffer.append(line.strip())
                    if len(chunk_buffer) >= chunk_size:
                        futures.append(
                            pool.submit(
                                _tokenize_chunk_worker,
                                chunk_buffer.copy(),
                                self.profile,
                                self.special_token_offset,
                                SPECIAL_TOKENS["<|eos|>"] if append_eos else None,
                            )
                        )
                        chunk_buffer.clear()
                
                # Submit remaining
                if chunk_buffer:
                    futures.append(
                        pool.submit(
                            _tokenize_chunk_worker,
                            chunk_buffer,
                            self.profile,
                            self.special_token_offset,
                            SPECIAL_TOKENS["<|eos|>"] if append_eos else None,
                        )
                    )
                
                # Collect and pack
                token_stream = []
                packed_sequences = []
                
                for future in as_completed(futures):
                    chunk_tokens = future.result()  # List[int]
                    token_stream.extend(chunk_tokens)
                    
                    # Pack into fixed-length sequences
                    while len(token_stream) >= seq_len:
                        packed_sequences.append(token_stream[:seq_len])
                        token_stream = token_stream[seq_len:]
                        total_tokens += seq_len
                    
                    # Flush to disk in chunks
                    if len(packed_sequences) >= 10_000:
                        _flush_to_disk(packed_sequences, output_path, total_tokens)
                        packed_sequences.clear()
                
                # Flush remaining
                if packed_sequences:
                    _flush_to_disk(packed_sequences, output_path, total_tokens)
        
        logger.info(f"Tokenization complete: {total_tokens:,} tokens written to {output_path}")
        return total_tokens
    
    def __repr__(self) -> str:
        return (f"ForgeTokenizer(profile={self.profile!r}, "
                f"vocab_size={self.vocab_size:,}, "
                f"n_workers={self.n_workers})")
    
    def save_pretrained(self, path: str):
        """Save tokenizer config for later loading."""
        import json
        os.makedirs(path, exist_ok=True)
        config = {
            "profile": self.profile,
            "device": self.device_str,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "special_token_offset": self.special_token_offset,
            "add_bos": self.add_bos,
            "add_eos": self.add_eos,
            "max_length": self.max_length,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Tokenizer config saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str) -> "ForgeTokenizer":
        import json
        with open(os.path.join(path, "tokenizer_config.json")) as f:
            config = json.load(f)
        return cls(
            profile=config["profile"],
            device=config.get("device", "cpu"),
            max_length=config.get("max_length"),
            add_bos=config.get("add_bos", True),
            add_eos=config.get("add_eos", True),
        )
    
    def __del__(self):
        """Graceful cleanup of thread pool."""
        try:
            self._thread_pool.shutdown(wait=False)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTIONS (must be module-level for pickling with ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_chunk_worker(
    texts: List[str],
    profile: str,
    offset: int,
    eos_id: Optional[int],
) -> List[int]:
    """
    Worker function for parallel tokenization.
    Runs in a separate process with its own CrayonVocab instance.
    """
    tok = CrayonVocab(device="cpu")
    tok.load_profile(profile)
    
    all_ids = []
    for text in texts:
        if not text:
            continue
        raw = tok.tokenize(text)
        if isinstance(raw, torch.Tensor):
            raw = raw.tolist()
        ids = [i + offset for i in raw]
        if eos_id is not None:
            ids.append(eos_id)
        all_ids.extend(ids)
    
    return all_ids


def _flush_to_disk(sequences: List[List[int]], output_path: str, offset: int):
    """Append packed sequences to a memory-mapped numpy file."""
    arr = np.array(sequences, dtype=np.uint32)
    mode = "ab"  # append binary
    with open(output_path + ".npy", mode) as f:
        np.save(f, arr)
    # Also write offset index for random access
    with open(output_path + ".idx", "ab") as f:
        np.save(f, np.array([offset], dtype=np.int64))


import math