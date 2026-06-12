"""
FORGE-3B Streaming Tokenizer.

A thin wrapper around ForgeTokenizer that streams a large corpus file and
writes tokenized output to disk as memory-mapped numpy arrays.

Designed for preprocessing pipelines where the full corpus does not fit in RAM.
Processes documents lazily in chunks, writing each completed shard atomically.

Usage:
    python -m tokenizer.streaming_tokenizer \\
        --input /data/raw/fineweb_edu \\
        --output /data/tokenized/fineweb_edu \\
        --profile standard \\
        --seq_len 2048 \\
        --shard_size 100000
"""

from __future__ import annotations

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Iterator, List

import numpy as np

logger = logging.getLogger(__name__)

_BOS   = 1
_EOS   = 2
_PAD   = 0
_DTYPE = np.uint32


# ─────────────────────────────────────────────────────────────────────────────
# Streaming Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

class StreamingTokenizer:
    """
    Tokenizes a corpus of JSONL files in a streaming fashion, packing
    tokens into fixed-length sequences and writing them to numbered .npy shards.

    Key properties:
      - O(shard_size * seq_len * 4 bytes) peak RAM usage — stream-safe
      - Atomic shard writes (temp file → rename)
      - Resumable: detects and skips existing shards
      - Thread-safe per instance (no shared state)
    """

    def __init__(
        self,
        tokenizer,
        output_dir: str,
        seq_len:    int = 2048,
        shard_size: int = 100_000,
        split:      str = "train",
    ):
        self.tok        = tokenizer
        self.seq_len    = seq_len
        self.shard_size = shard_size
        self.split      = split

        self._out_dir = Path(output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

        # Rolling packed-token buffer
        self._buffer: List[int] = []

        # Shard accumulation buffer
        self._shard_seqs: List[np.ndarray] = []
        self._shard_idx = self._detect_next_shard()

        # Stats
        self._n_docs   = 0
        self._n_tokens = 0
        self._n_seqs   = 0
        self._t0       = time.perf_counter()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_next_shard(self) -> int:
        """Skip past shards that already exist on disk (resumable run)."""
        existing = sorted(self._out_dir.glob(f"{self.split}_shard_*.npy"))
        if not existing:
            return 0
        idx = int(existing[-1].stem.split("_")[-1])
        logger.info(f"StreamingTokenizer: resuming after shard {idx}")
        return idx + 1

    def _shard_path(self, idx: int) -> Path:
        return self._out_dir / f"{self.split}_shard_{idx:04d}.npy"

    def _flush_shard(self):
        """Write the shard buffer to disk atomically via temp→rename."""
        if not self._shard_seqs:
            return
        arr  = np.stack(self._shard_seqs, axis=0)   # (N, seq_len) uint32
        path = self._shard_path(self._shard_idx)
        tmp  = path.with_suffix(".tmp.npy")
        np.save(str(tmp), arr)
        tmp.rename(path)
        logger.info(
            f"  Shard {self._shard_idx:04d}: {len(arr)} seqs → {path.name} "
            f"({arr.nbytes / 1e6:.1f} MB)"
        )
        self._shard_idx  += 1
        self._shard_seqs  = []

    def _drain_buffer(self):
        """Carve complete sequences from the rolling buffer."""
        while len(self._buffer) >= self.seq_len:
            seq = np.array(self._buffer[: self.seq_len], dtype=_DTYPE)
            self._buffer    = self._buffer[self.seq_len :]
            self._shard_seqs.append(seq)
            self._n_seqs   += 1

            if len(self._shard_seqs) >= self.shard_size:
                self._flush_shard()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_document(self, text: str):
        """Tokenize one text document and pack its tokens into the buffer."""
        try:
            token_ids = self.tok.encode(
                text,
                add_bos=False,
                add_eos=False,
                truncate=False,
            )
        except Exception as exc:
            logger.warning(f"Tokenization failed: {exc} — skipping document")
            return

        if not token_ids:
            return

        # BOS + tokens + EOS
        self._buffer.extend([_BOS] + list(token_ids) + [_EOS])
        self._n_tokens += len(token_ids) + 2
        self._n_docs   += 1
        self._drain_buffer()

    def add_token_array(self, token_ids: np.ndarray):
        """Add a pre-tokenized uint32 array directly (bypasses the tokenizer)."""
        self._buffer.extend([_BOS] + token_ids.tolist() + [_EOS])
        self._n_tokens += len(token_ids) + 2
        self._n_docs   += 1
        self._drain_buffer()

    def finalize(self) -> dict:
        """
        Flush all remaining sequences (partial last shard) and write metadata.
        Must be called exactly once after all documents have been added.
        Returns a summary dict.
        """
        # Drain any remaining full sequences
        if len(self._buffer) >= self.seq_len:
            self._drain_buffer()

        # Write partial shard if there is anything left
        if self._shard_seqs:
            self._flush_shard()

        elapsed = time.perf_counter() - self._t0
        speed   = self._n_tokens / max(elapsed, 1e-6)

        meta = {
            "n_documents":    self._n_docs,
            "n_tokens":       self._n_tokens,
            "n_sequences":    self._n_seqs,
            "seq_len":        self.seq_len,
            "shard_size":     self.shard_size,
            "n_shards":       self._shard_idx,
            "elapsed_s":      round(elapsed, 1),
            "tokens_per_sec": int(speed),
        }

        meta_path = self._out_dir / f"{self.split}_streaming_meta.json"
        with open(str(meta_path), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(
            f"StreamingTokenizer finalized: "
            f"{self._n_seqs:,} seqs | "
            f"{self._n_tokens / 1e9:.2f}B tokens | "
            f"{speed / 1e6:.1f}M tok/s | "
            f"{elapsed:.1f}s"
        )
        return meta


# ─────────────────────────────────────────────────────────────────────────────
# Corpus Iterator
# ─────────────────────────────────────────────────────────────────────────────

def _iter_texts(input_path: Path, text_key: str = "text") -> Iterator[str]:
    """Yield text strings from all JSONL / plain-text files under input_path."""
    files: list = []
    for ext in (".jsonl", ".json", ".txt"):
        files.extend(sorted(input_path.rglob(f"*{ext}")))

    if not files:
        raise FileNotFoundError(f"No JSONL/txt files found under {input_path}")

    logger.info(f"_iter_texts: found {len(files)} files")

    for path in files:
        try:
            if path.suffix == ".txt":
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    yield text
            else:
                with open(str(path), encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj  = json.loads(line)
                            text = obj.get(text_key, "")
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            continue
        except OSError as exc:
            logger.warning(f"Cannot read {path}: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="FORGE-3B Streaming Tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      required=True, help="Input dir with JSONL/txt files")
    parser.add_argument("--output",     required=True, help="Output dir for .npy shards")
    parser.add_argument("--profile",    default="standard", choices=["standard", "lite"])
    parser.add_argument("--seq_len",    type=int, default=2048)
    parser.add_argument("--shard_size", type=int, default=100_000)
    parser.add_argument("--split",      default="train")
    parser.add_argument("--text_key",   default="text")
    parser.add_argument("--n_workers",  type=int,
                        default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    from tokenizer.crayon_wrapper import ForgeTokenizer

    tok = ForgeTokenizer(
        profile=args.profile,
        device="cpu",
        n_workers=args.n_workers,
        max_length=args.seq_len * 8,
    )

    streamer = StreamingTokenizer(
        tokenizer=tok,
        output_dir=args.output,
        seq_len=args.seq_len,
        shard_size=args.shard_size,
        split=args.split,
    )

    input_path = Path(args.input)
    n = 0
    for text in _iter_texts(input_path, text_key=args.text_key):
        streamer.add_document(text)
        n += 1
        if n % 100_000 == 0:
            logger.info(f"Processed {n:,} documents...")

    meta = streamer.finalize()
    print(json.dumps(meta, indent=2))
