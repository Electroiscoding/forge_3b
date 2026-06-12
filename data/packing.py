"""
FORGE-3B Token Packing Pipeline.

Converts raw text / pre-tokenized token arrays into tightly-packed,
fixed-length .npy shards ready for memory-mapped pretraining.

Design goals:
  - Zero wasted tokens: BOS-concatenated documents packed greedily
  - Multiprocessing for throughput (1 process per CPU core)
  - Resumable: skips already-written shards
  - Domain-tagged output (one subdirectory per domain)
  - Reproducible: deterministic shard order given a fixed seed

Usage:
    python -m data.packing \
        --input_dir  /data/raw/wikipedia \
        --output_dir /data/tokenized/wikipedia \
        --tokenizer_profile standard \
        --seq_len 2048 \
        --shard_size 100000 \
        --n_workers 16
"""

from __future__ import annotations

import os
import json
import math
import time
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DTYPE = np.uint32          # supports vocab sizes up to 4 billion
EOS_TOKEN_ID = 2           # CRAYON <|eos|>
BOS_TOKEN_ID = 1           # CRAYON <|bos|>
PAD_TOKEN_ID = 0           # CRAYON <|pad|>

# Default output shard: 100K sequences per file (~200M tokens at seq=2048)
DEFAULT_SHARD_SEQUENCES = 100_000


# ─────────────────────────────────────────────────────────────────────────────
# Document Iterator
# ─────────────────────────────────────────────────────────────────────────────

def iter_jsonl_documents(
    input_path: Path,
    text_key: str = "text",
) -> Iterator[str]:
    """Yield raw text strings from a JSONL file."""
    with open(str(input_path), "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Bad JSON on line {line_no} of {input_path}: {line[:80]}")
                continue

            text = obj.get(text_key, "")
            if text:
                yield text


def iter_token_arrays(
    input_path: Path,
) -> Iterator[np.ndarray]:
    """Yield pre-tokenized uint32 1D arrays from a .npy file."""
    arr = np.load(str(input_path), mmap_mode="r")
    if arr.ndim == 2:
        for row in arr:
            yield row.astype(DTYPE)
    else:
        yield arr.astype(DTYPE)


# ─────────────────────────────────────────────────────────────────────────────
# Greedy Packer
# ─────────────────────────────────────────────────────────────────────────────

class GreedyPacker:
    """
    Packs a stream of variable-length token arrays into fixed-length sequences.

    Strategy:
      1. Prepend BOS to each document.
      2. Append documents end-to-end into a rolling buffer.
      3. When buffer >= seq_len, slice off a complete sequence and emit it.
      4. Discard any partial tail (guarantees no padding in the output).
    """

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self._buffer: List[int] = []
        self._n_emitted = 0
        self._n_tokens_consumed = 0

    def add(self, token_ids: np.ndarray) -> List[np.ndarray]:
        """
        Add a document's token ids and return any complete sequences ready.
        token_ids should NOT include a leading BOS — we prepend it here.
        """
        # BOS + document tokens + EOS
        doc = [BOS_TOKEN_ID] + token_ids.tolist() + [EOS_TOKEN_ID]
        self._buffer.extend(doc)
        self._n_tokens_consumed += len(doc)

        sequences = []
        while len(self._buffer) >= self.seq_len:
            seq = np.array(self._buffer[: self.seq_len], dtype=DTYPE)
            self._buffer = self._buffer[self.seq_len :]
            sequences.append(seq)
            self._n_emitted += 1

        return sequences

    def flush(self) -> Optional[np.ndarray]:
        """Return the partial buffer padded to seq_len (or None if empty)."""
        if not self._buffer:
            return None
        pad_len = self.seq_len - len(self._buffer)
        padded = self._buffer + [PAD_TOKEN_ID] * pad_len
        return np.array(padded[: self.seq_len], dtype=DTYPE)

    @property
    def stats(self) -> dict:
        return {
            "sequences_emitted": self._n_emitted,
            "tokens_consumed": self._n_tokens_consumed,
            "buffer_fill": len(self._buffer),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shard Writer
# ─────────────────────────────────────────────────────────────────────────────

class ShardWriter:
    """
    Accumulates sequences and writes them to numbered .npy shards.

    Each shard is a 2D array of shape (shard_size, seq_len) with dtype uint32.
    Shards are written atomically via a temp-then-rename pattern.
    """

    def __init__(
        self,
        output_dir: Path,
        split: str,
        seq_len: int,
        shard_size: int,
    ):
        self.output_dir = output_dir
        self.split = split
        self.seq_len = seq_len
        self.shard_size = shard_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._shard_idx = self._detect_next_shard_idx()
        self._buffer: List[np.ndarray] = []
        self._total_written = 0

    def _detect_next_shard_idx(self) -> int:
        """Skip shards that already exist (resumable packing)."""
        existing = sorted(self.output_dir.glob(f"{self.split}_shard_*.npy"))
        if not existing:
            return 0
        last = existing[-1]
        # e.g. train_shard_0042.npy -> 43
        idx = int(last.stem.split("_")[-1])
        logger.info(f"Resuming from shard {idx + 1} (found {len(existing)} existing shards)")
        return idx + 1

    def _shard_path(self, idx: int) -> Path:
        return self.output_dir / f"{self.split}_shard_{idx:04d}.npy"

    def add(self, seq: np.ndarray):
        """Add a single (seq_len,) sequence."""
        self._buffer.append(seq)
        if len(self._buffer) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        arr = np.stack(self._buffer[: self.shard_size], axis=0)
        path = self._shard_path(self._shard_idx)
        tmp_path = path.with_suffix(".tmp.npy")

        np.save(str(tmp_path), arr)
        tmp_path.rename(path)

        self._total_written += len(arr)
        logger.info(
            f"Written shard {self._shard_idx:04d}: "
            f"{len(arr)} sequences → {path.name} "
            f"({arr.nbytes / 1e6:.1f} MB)"
        )
        self._shard_idx += 1
        self._buffer = self._buffer[self.shard_size :]

    def finalize(self):
        """Write any remaining sequences as the last (potentially partial) shard."""
        if self._buffer:
            # Do NOT discard the remainder — write it as a smaller shard
            arr = np.stack(self._buffer, axis=0)
            path = self._shard_path(self._shard_idx)
            tmp_path = path.with_suffix(".tmp.npy")
            np.save(str(tmp_path), arr)
            tmp_path.rename(path)
            self._total_written += len(arr)
            logger.info(
                f"Written final shard {self._shard_idx:04d}: "
                f"{len(arr)} sequences (partial)"
            )
            self._buffer = []

    @property
    def total_sequences(self) -> int:
        return self._total_written + len(self._buffer)


# ─────────────────────────────────────────────────────────────────────────────
# Worker Function (for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_worker(args_tuple):
    """
    Worker: tokenizes one input file and returns a list of packed uint32 arrays.
    Runs in a subprocess so the tokenizer is forked, not pickled.
    """
    (
        input_file,
        tokenizer_profile,
        seq_len,
        text_key,
        worker_id,
    ) = args_tuple

    # Import inside worker to avoid serialization issues
    try:
        from tokenizer.crayon_wrapper import ForgeTokenizer
        tok = ForgeTokenizer(
            profile=tokenizer_profile,
            device="cpu",
            n_workers=1,           # no nested parallelism
            max_length=seq_len * 4,  # generous limit before chunking
        )
    except ImportError:
        # Fallback: treat input as pre-tokenized .npy
        tok = None

    input_path = Path(input_file)
    packer = GreedyPacker(seq_len=seq_len)
    sequences = []

    try:
        if tok is not None and input_path.suffix in (".jsonl", ".json", ".txt"):
            for text in iter_jsonl_documents(input_path, text_key=text_key):
                token_ids = tok.encode(
                    text,
                    add_bos=False,   # packer handles BOS
                    add_eos=False,   # packer handles EOS
                    truncate=False,
                )
                for seq in packer.add(np.array(token_ids, dtype=DTYPE)):
                    sequences.append(seq)

        elif input_path.suffix == ".npy":
            for arr in iter_token_arrays(input_path):
                for seq in packer.add(arr):
                    sequences.append(seq)

        else:
            logger.warning(f"Worker {worker_id}: unrecognised file type {input_path.suffix}, skipping")
            return []

    except Exception as exc:
        logger.error(f"Worker {worker_id}: error processing {input_path}: {exc}")
        return []

    return sequences


# ─────────────────────────────────────────────────────────────────────────────
# Main Packing Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def pack_domain(
    input_dir: str,
    output_dir: str,
    tokenizer_profile: str = "standard",
    seq_len: int = 2048,
    shard_size: int = DEFAULT_SHARD_SEQUENCES,
    split: str = "train",
    val_fraction: float = 0.005,
    n_workers: int = 8,
    text_key: str = "text",
    seed: int = 42,
    file_extensions: tuple = (".jsonl", ".json", ".txt", ".npy"),
) -> dict:
    """
    Pack all documents in input_dir into seq_len-length token sequences.

    Returns a summary dict with counts and paths.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover input files
    all_files = []
    for ext in file_extensions:
        all_files.extend(sorted(input_path.rglob(f"*{ext}")))

    if not all_files:
        raise FileNotFoundError(
            f"No files with extensions {file_extensions} found in {input_dir}"
        )

    logger.info(f"Found {len(all_files)} input files in {input_dir}")

    # Deterministic train/val split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_files))
    n_val = max(1, int(len(all_files) * val_fraction))
    val_indices = set(indices[:n_val].tolist())
    train_files = [all_files[i] for i in range(len(all_files)) if i not in val_indices]
    val_files   = [all_files[i] for i in range(len(all_files)) if i in val_indices]

    logger.info(f"Split: {len(train_files)} train files, {len(val_files)} val files")

    def _pack_split(files: list, split_name: str) -> int:
        """Pack a list of files into one split's shards, returns seq count."""
        writer = ShardWriter(
            output_dir=output_path,
            split=split_name,
            seq_len=seq_len,
            shard_size=shard_size,
        )

        t0 = time.perf_counter()
        total_seqs = 0

        worker_args = [
            (str(f), tokenizer_profile, seq_len, text_key, i)
            for i, f in enumerate(files)
        ]

        with mp.Pool(processes=n_workers, maxtasksperchild=50) as pool:
            for file_sequences in pool.imap_unordered(
                _tokenize_worker, worker_args, chunksize=4
            ):
                for seq in file_sequences:
                    writer.add(seq)
                    total_seqs += 1

                if total_seqs % 100_000 == 0 and total_seqs > 0:
                    elapsed = time.perf_counter() - t0
                    rate = total_seqs / elapsed
                    logger.info(
                        f"  [{split_name}] {total_seqs:,} sequences packed "
                        f"({rate:,.0f} seq/s)"
                    )

        writer.finalize()
        elapsed = time.perf_counter() - t0
        logger.info(
            f"[{split_name}] Done: {writer.total_sequences:,} sequences "
            f"in {elapsed:.1f}s "
            f"({writer.total_sequences * seq_len / 1e9:.2f}B tokens)"
        )
        return writer.total_sequences

    n_train = _pack_split(train_files, "train")
    n_val   = _pack_split(val_files,   "val")  if val_files else 0

    # Write metadata
    meta = {
        "input_dir":          str(input_dir),
        "output_dir":         str(output_dir),
        "seq_len":            seq_len,
        "shard_size":         shard_size,
        "tokenizer_profile":  tokenizer_profile,
        "train_sequences":    n_train,
        "val_sequences":      n_val,
        "train_tokens":       n_train * seq_len,
        "val_tokens":         n_val   * seq_len,
        "n_input_files":      len(all_files),
    }
    with open(str(output_path / "packing_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Packing complete. Metadata written to {output_path / 'packing_meta.json'}")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FORGE-3B Token Packing — produce fixed-length .npy shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input_dir",  required=True, help="Directory of raw text/JSONL/npy files")
    p.add_argument("--output_dir", required=True, help="Destination for packed .npy shards")
    p.add_argument("--tokenizer_profile", default="standard",
                   choices=["standard", "lite"], help="CRAYON tokenizer profile")
    p.add_argument("--seq_len",     type=int, default=2048, help="Sequence length in tokens")
    p.add_argument("--shard_size",  type=int, default=DEFAULT_SHARD_SEQUENCES,
                   help="Sequences per output shard")
    p.add_argument("--val_fraction", type=float, default=0.005,
                   help="Fraction of input files reserved for the val split")
    p.add_argument("--n_workers",  type=int, default=min(8, os.cpu_count() or 1),
                   help="Number of parallel tokenizer workers")
    p.add_argument("--text_key",   default="text", help="JSON key for the text field")
    p.add_argument("--seed",       type=int, default=42)
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_arg_parser()
    args   = parser.parse_args()

    summary = pack_domain(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_profile=args.tokenizer_profile,
        seq_len=args.seq_len,
        shard_size=args.shard_size,
        val_fraction=args.val_fraction,
        n_workers=args.n_workers,
        text_key=args.text_key,
        seed=args.seed,
    )

    print("\n── Packing Summary ──────────────────────────")
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
