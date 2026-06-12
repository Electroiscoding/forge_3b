"""
FORGE-3B Data Preprocessing Pipeline.

Orchestrates the full raw-data → tokenized-shards pipeline:
  1. Download / verify source data (or accept local paths)
  2. Filter quality (language detection, min-length, dedup hashing)
  3. Tokenize with CRAYON
  4. Pack into fixed-length shards via data.packing
  5. Write per-domain packing_meta.json

Designed to be run ONCE before pretraining and then never again.

Usage (single domain):
    python -m data.preprocessing \
        --domain wikipedia \
        --raw_dir /data/raw/wikipedia \
        --out_dir /data/tokenized/wikipedia \
        --seq_len 2048 \
        --n_workers 32

Usage (all domains at once):
    python -m data.preprocessing --all \
        --raw_base  /data/raw \
        --out_base  /data/tokenized \
        --n_workers 32
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, Optional, Set

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Quality Filter
# ─────────────────────────────────────────────────────────────────────────────

# Heuristic thresholds aligned with FineWeb-Edu / Dolma quality pipeline
_MIN_CHARS          = 150          # discard very short documents
_MAX_CHARS          = 1_000_000    # discard suspiciously large blobs
_MIN_ALPHA_RATIO    = 0.60         # at least 60% alphabetic chars
_MAX_SYMBOL_RATIO   = 0.10         # at most 10% non-alphanumeric non-space chars
_MAX_LINE_NEWLINE   = 0.30         # discard if >30% of chars are newlines


def is_quality_document(text: str) -> bool:
    """
    Fast, rule-based quality check. Returns True if the document passes.
    Applied BEFORE tokenization to avoid wasting tokenizer time on garbage.
    """
    n = len(text)

    if n < _MIN_CHARS or n > _MAX_CHARS:
        return False

    alpha   = sum(c.isalpha() for c in text)
    newline = text.count("\n")
    symbols = sum(not (c.isalnum() or c.isspace()) for c in text)

    if alpha / n < _MIN_ALPHA_RATIO:
        return False
    if newline / n > _MAX_LINE_NEWLINE:
        return False
    if symbols / n > _MAX_SYMBOL_RATIO:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Exact-Duplicate Filter (MinHash-lite: URL-level and first-512-char hash)
# ─────────────────────────────────────────────────────────────────────────────

class DedupFilter:
    """
    Lightweight exact-dedup using SHA-256 of the first 512 characters.
    For fuzzy dedup at scale, use the separate dedup_minhash.py script.
    """

    def __init__(self, seen_hashes: Optional[Set[str]] = None):
        self._seen: Set[str] = seen_hashes or set()

    def is_duplicate(self, text: str) -> bool:
        snippet = text[:512].strip()
        h = hashlib.sha256(snippet.encode("utf-8", errors="replace")).hexdigest()
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

    @property
    def n_seen(self) -> int:
        return len(self._seen)


# ─────────────────────────────────────────────────────────────────────────────
# Document Iterator (raw JSONL / text)
# ─────────────────────────────────────────────────────────────────────────────

def iter_documents(
    raw_dir: Path,
    text_key: str = "text",
    extensions: tuple = (".jsonl", ".json", ".txt"),
) -> Iterator[str]:
    """
    Recursively yield text strings from all matching files in raw_dir.
    Plain .txt files yield the entire file as one document.
    """
    files = []
    for ext in extensions:
        files.extend(sorted(raw_dir.rglob(f"*{ext}")))

    if not files:
        raise FileNotFoundError(f"No files with extensions {extensions} in {raw_dir}")

    logger.info(f"iter_documents: found {len(files)} files in {raw_dir}")

    for path in files:
        try:
            if path.suffix == ".txt":
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    yield text
            else:
                with open(str(path), "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = obj.get(text_key, "")
                        if text:
                            yield text
        except OSError as exc:
            logger.warning(f"Cannot read {path}: {exc}")
            continue


# ─────────────────────────────────────────────────────────────────────────────
# Domain-level Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_domain(
    domain: str,
    raw_dir: str,
    out_dir: str,
    tokenizer_profile: str = "standard",
    seq_len: int = 2048,
    shard_size: int = 100_000,
    val_fraction: float = 0.005,
    n_workers: int = 8,
    text_key: str = "text",
    apply_quality_filter: bool = True,
    apply_dedup: bool = True,
    seed: int = 42,
) -> dict:
    """
    Full preprocessing pipeline for one domain:
      read → quality-filter → dedup → pack → shard

    Returns a summary dict.
    """
    from data.packing import pack_domain

    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ── Phase 1: Filter & write a clean intermediate JSONL ───────────────────
    clean_jsonl = out_path / "_clean.jsonl"

    if clean_jsonl.exists():
        logger.info(f"[{domain}] Reusing existing clean JSONL: {clean_jsonl}")
    else:
        logger.info(f"[{domain}] Filtering raw documents from {raw_path}...")
        dedup = DedupFilter() if apply_dedup else None

        n_seen = 0
        n_passed = 0
        n_quality_fail = 0
        n_dup = 0

        with open(str(clean_jsonl), "w", encoding="utf-8") as out_f:
            for text in iter_documents(raw_path, text_key=text_key):
                n_seen += 1

                if apply_quality_filter and not is_quality_document(text):
                    n_quality_fail += 1
                    continue

                if dedup is not None and dedup.is_duplicate(text):
                    n_dup += 1
                    continue

                out_f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n_passed += 1

                if n_seen % 500_000 == 0:
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        f"[{domain}] Filtered {n_seen:,} docs | "
                        f"passed={n_passed:,} | "
                        f"quality_fail={n_quality_fail:,} | "
                        f"dup={n_dup:,} | "
                        f"{elapsed:.0f}s"
                    )

        elapsed = time.perf_counter() - t0
        logger.info(
            f"[{domain}] Filter complete: {n_passed:,}/{n_seen:,} docs passed "
            f"({100*n_passed/max(1,n_seen):.1f}%) in {elapsed:.1f}s"
        )

    # ── Phase 2: Tokenise + Pack ─────────────────────────────────────────────
    logger.info(f"[{domain}] Packing to seq_len={seq_len}...")

    meta = pack_domain(
        input_dir=str(clean_jsonl.parent),   # packing scans .jsonl in this dir
        output_dir=str(out_path),
        tokenizer_profile=tokenizer_profile,
        seq_len=seq_len,
        shard_size=shard_size,
        val_fraction=val_fraction,
        n_workers=n_workers,
        text_key="text",                      # clean JSONL always uses "text"
        seed=seed,
    )

    # Cleanup intermediate file to save disk space
    try:
        clean_jsonl.unlink()
    except OSError:
        pass

    elapsed = time.perf_counter() - t0
    meta["domain"]   = domain
    meta["elapsed_s"] = round(elapsed, 1)
    logger.info(f"[{domain}] Total preprocessing time: {elapsed:.1f}s")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Domain Orchestration
# ─────────────────────────────────────────────────────────────────────────────

# Canonical domain list matching the README data table
ALL_DOMAINS = [
    "fineweb_edu",
    "thestack",
    "wikipedia",
    "openwebmath",
    "books",
    "arxiv",
    "dolma",
    "stackexchange",
    "redpajama_cc",
    "multilingual",
]

# Per-domain text-key overrides (most are just "text")
DOMAIN_TEXT_KEYS = {
    "thestack":      "content",
    "arxiv":         "text",
    "stackexchange": "body",
}

# Per-domain quality-filter overrides (code domains: relax alpha ratio)
DOMAIN_QUALITY_FILTER = {
    "thestack":  False,    # code has low alpha ratio
    "openwebmath": True,
}


def preprocess_all(
    raw_base: str,
    out_base: str,
    tokenizer_profile: str = "standard",
    seq_len: int = 2048,
    shard_size: int = 100_000,
    n_workers: int = 8,
    seed: int = 42,
) -> None:
    """
    Run preprocessing for ALL domains sequentially.
    Each domain is independent — safe to run in parallel via tmux if needed.
    """
    raw_base_path = Path(raw_base)
    out_base_path = Path(out_base)

    summaries = []
    for domain in ALL_DOMAINS:
        raw_dir = str(raw_base_path / domain)
        out_dir = str(out_base_path / domain)

        if not Path(raw_dir).exists():
            logger.warning(f"Domain '{domain}' raw dir missing: {raw_dir} — skipping")
            continue

        text_key     = DOMAIN_TEXT_KEYS.get(domain, "text")
        quality_on   = DOMAIN_QUALITY_FILTER.get(domain, True)

        logger.info(f"\n{'='*60}")
        logger.info(f" Domain: {domain.upper()}")
        logger.info(f"{'='*60}")

        try:
            meta = preprocess_domain(
                domain=domain,
                raw_dir=raw_dir,
                out_dir=out_dir,
                tokenizer_profile=tokenizer_profile,
                seq_len=seq_len,
                shard_size=shard_size,
                n_workers=n_workers,
                text_key=text_key,
                apply_quality_filter=quality_on,
                apply_dedup=True,
                seed=seed,
            )
            summaries.append(meta)
        except Exception as exc:
            logger.error(f"Domain '{domain}' FAILED: {exc}", exc_info=True)
            summaries.append({"domain": domain, "error": str(exc)})

    # Write global manifest
    manifest_path = out_base_path / "preprocessing_manifest.json"
    with open(str(manifest_path), "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info(f"\nManifest written: {manifest_path}")

    # Print table
    print("\n── Preprocessing Summary ───────────────────────────────────────")
    print(f"{'Domain':<20} {'Train Seq':>12} {'Val Seq':>10} {'Train Tokens (B)':>18}")
    print("─" * 65)
    for m in summaries:
        if "error" in m:
            print(f"{m.get('domain','?'):<20}  ERROR: {m['error'][:40]}")
        else:
            print(
                f"{m.get('domain','?'):<20} "
                f"{m.get('train_sequences',0):>12,} "
                f"{m.get('val_sequences',0):>10,} "
                f"{m.get('train_tokens',0)/1e9:>18.2f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FORGE-3B Data Preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--domain", help="Process a single named domain")
    mode.add_argument("--all",    action="store_true", help="Process all domains")

    # Single-domain args
    p.add_argument("--raw_dir", help="Raw input dir (single domain)")
    p.add_argument("--out_dir", help="Output dir (single domain)")

    # Multi-domain args
    p.add_argument("--raw_base", default="/data/raw",       help="Raw data root (--all)")
    p.add_argument("--out_base", default="/data/tokenized", help="Output root (--all)")

    # Shared args
    p.add_argument("--tokenizer_profile", default="standard", choices=["standard", "lite"])
    p.add_argument("--seq_len",    type=int, default=2048)
    p.add_argument("--shard_size", type=int, default=100_000)
    p.add_argument("--n_workers",  type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--text_key",   default="text")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--no_quality_filter", action="store_true")
    p.add_argument("--no_dedup",          action="store_true")
    return p


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_arg_parser()
    args   = parser.parse_args()

    if args.all:
        preprocess_all(
            raw_base=args.raw_base,
            out_base=args.out_base,
            tokenizer_profile=args.tokenizer_profile,
            seq_len=args.seq_len,
            shard_size=args.shard_size,
            n_workers=args.n_workers,
            seed=args.seed,
        )
    else:
        if not args.raw_dir or not args.out_dir:
            parser.error("--raw_dir and --out_dir are required with --domain")
        meta = preprocess_domain(
            domain=args.domain,
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            tokenizer_profile=args.tokenizer_profile,
            seq_len=args.seq_len,
            shard_size=args.shard_size,
            n_workers=args.n_workers,
            text_key=args.text_key,
            apply_quality_filter=not args.no_quality_filter,
            apply_dedup=not args.no_dedup,
            seed=args.seed,
        )
        print(json.dumps(meta, indent=2))
