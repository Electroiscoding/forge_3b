"""
FORGE-3B Kaggle Data Pipeline — Final Production Version.

Streams all 10 pretraining domains from HuggingFace, quality-filters, tokenizes
with CRAYON, packs into fixed-length .npy shards, uploads to HuggingFace Hub,
and verifies ≥50B total tokens.

Target: Kaggle free tier (no GPU, 4 CPU, ~30 GB RAM, ~70 GB disk). Runtime ≤ 10h.

Usage (Kaggle notebook — paste as one cell):
    ───────────────────────────────────────────────
    !pip install -q datasets xerv-crayon torch numpy huggingface_hub
    !git clone https://github.com/Electroiscoding/forge_3b.git
    %cd forge_3b
    !python kaggle_data_pipeline.py
    ───────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import gc
import json
import time
import math
import shutil
import hashlib
import logging
import argparse
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

logger = logging.getLogger("forge_pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Retrieve credentials from environment or Kaggle UserSecrets to prevent committing secrets to git
HF_USERNAME  = os.environ.get("HF_USERNAME", "Phase-Technologies")
HF_TOKEN     = os.environ.get("HF_TOKEN")

# Fallback: Check if running inside Kaggle and load token from Kaggle UserSecrets
if not HF_TOKEN:
    try:
        # Use dynamic import to prevent static linters from failing in non-Kaggle environments
        kaggle_secrets = __import__("kaggle_secrets")
        UserSecretsClient = kaggle_secrets.UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

# A placeholder if not set
if not HF_TOKEN:
    HF_TOKEN = ""

HF_REPO_ID   = f"{HF_USERNAME}/forge-3b-pretrain-data"
OUT_DIR      = "/kaggle/working/data"

# Token packing constants
BOS_ID = 1
EOS_ID = 2
PAD_ID = 0
DTYPE  = np.uint32

# Quality filter thresholds (FineWeb-Edu / Dolma aligned)
_MIN_CHARS        = 150
_MAX_CHARS        = 1_000_000
_MIN_ALPHA_RATIO  = 0.55
_MAX_SYMBOL_RATIO = 0.12
_MAX_NEWLINE      = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# Dataset Registry — ALL 10 domains, verified HF sources
#
# target_docs are calibrated with ~25% overshoot above the minimum needed,
# to account for quality-filter + dedup losses and still guarantee ≥50B tokens.
# ─────────────────────────────────────────────────────────────────────────────

DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fineweb_edu": {
        "hf_id":          "HuggingFaceFW/fineweb-edu",
        "hf_config":      "CC-MAIN-2024-10",
        "split":          "train",
        "text_key":       "text",
        "quality_filter": True,
        "target_docs":    9_000_000,    # 30% of 50B = 15B tokens, ~2K tok/doc avg
        "target_tokens":  15_000_000_000,
        "weight_pct":     30,
    },
    "thestack": {
        "hf_id":          "bigcode/starcoderdata",
        "hf_config":      None,
        "hf_data_dir":    "python",
        "split":          "train",
        "text_key":       "content",
        "quality_filter": False,
        "target_docs":    5_000_000,    # 16% = 8B tokens
        "target_tokens":  8_000_000_000,
        "weight_pct":     16,
    },
    "wikipedia": {
        "hf_id":          "wikimedia/wikipedia",
        "hf_config":      "20231101.en",
        "split":          "train",
        "text_key":       "text",
        "quality_filter": True,
        "target_docs":    7_000_000,    # 8% = 4B tokens, wiki articles are short
        "target_tokens":  4_000_000_000,
        "weight_pct":     8,
    },
    "openwebmath": {
        "hf_id":          "open-web-math/open-web-math",
        "hf_config":      None,
        "split":          "train",
        "text_key":       "text",
        "quality_filter": True,
        "target_docs":    2_500_000,    # 8% = 4B tokens
        "target_tokens":  4_000_000_000,
        "weight_pct":     8,
    },
    "books": {
        "hf_id":          "emozilla/pg19",
        "hf_config":      None,
        "split":          "train",
        "text_key":       "text",
        "quality_filter": False,
        "target_docs":    30_000,       # 7% = 3.5B tokens, books are very long
        "target_tokens":  3_500_000_000,
        "weight_pct":     7,
    },
    "arxiv": {
        "hf_id":          "common-pile/arxiv_papers",
        "hf_config":      None,
        "split":          "train",
        "text_key":       "text",
        "quality_filter": False,
        "target_docs":    350_000,      # 6% = 3B tokens, papers are long
        "target_tokens":  3_000_000_000,
        "weight_pct":     6,
    },
    "dolma": {
        "hf_id":          "allenai/dolma",
        "hf_config":      "v1_7",
        "split":          "train",
        "text_key":       "text",
        "quality_filter": True,
        "target_docs":    3_000_000,    # 10% = 5B tokens
        "target_tokens":  5_000_000_000,
        "weight_pct":     10,
    },
    "stackexchange": {
        "hf_id":          "flax-sentence-embeddings/stackexchange_title_body_jsonl",
        "hf_config":      None,
        "split":          "train",
        "text_key":       "title_body",
        "quality_filter": True,
        "target_docs":    2_000_000,    # 5% = 2.5B tokens
        "target_tokens":  2_500_000_000,
        "weight_pct":     5,
    },
    "redpajama_cc": {
        "hf_id":          "togethercomputer/RedPajama-Data-V2",
        "hf_config":      "sample",
        "split":          "train",
        "text_key":       "raw_content",
        "quality_filter": True,
        "target_docs":    2_000_000,    # 6% = 3B tokens
        "target_tokens":  3_000_000_000,
        "weight_pct":     6,
    },
    "multilingual": {
        "hf_id":          "allenai/c4",
        "hf_config":      None,
        "split":          "train",
        "text_key":       "text",
        "quality_filter": True,
        "target_docs":    1_200_000,    # 4% = 2B tokens
        "target_tokens":  2_000_000_000,
        "langs":          ["de", "fr", "es", "zh", "ja", "ru", "pt", "it", "nl", "ar"],
        "weight_pct":     4,
    },
}

TOTAL_TARGET_TOKENS = 50_000_000_000  # 50B


# ─────────────────────────────────────────────────────────────────────────────
# Quality Filter
# ─────────────────────────────────────────────────────────────────────────────

def is_quality_document(text: str) -> bool:
    """Fast rule-based quality check. Samples first 2K chars for speed."""
    n = len(text)
    if n < _MIN_CHARS or n > _MAX_CHARS:
        return False
    sample = min(n, 2000)
    snippet = text[:sample]
    alpha   = sum(c.isalpha() for c in snippet)
    newline = snippet.count("\n")
    symbols = sum(not (c.isalnum() or c.isspace()) for c in snippet)
    if alpha / sample < _MIN_ALPHA_RATIO:
        return False
    if newline / sample > _MAX_NEWLINE:
        return False
    if symbols / sample > _MAX_SYMBOL_RATIO:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Dedup Filter
# ─────────────────────────────────────────────────────────────────────────────

class DedupFilter:
    """SHA-256 exact-dedup on first 512 chars."""
    def __init__(self):
        self._seen = set()

    def is_duplicate(self, text: str) -> bool:
        h = hashlib.sha256(text[:512].encode("utf-8", errors="replace")).hexdigest()
        if h in self._seen:
            return True
        self._seen.add(h)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Inline Packer + Shard Writer
# ─────────────────────────────────────────────────────────────────────────────

class InlinePacker:
    """
    Streams token IDs → packs into fixed-length sequences → writes .npy shards.
    No intermediate files. Memory bounded by shard_size.
    """

    def __init__(self, output_dir: Path, seq_len: int = 2048,
                 shard_size: int = 50_000, split: str = "train"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len    = seq_len
        self.shard_size = shard_size
        self.split      = split
        self._buffer: List[int] = []
        self._shard_seqs: List[np.ndarray] = []
        self._shard_idx  = 0
        self._n_seqs     = 0
        self._n_tokens   = 0
        self._n_docs     = 0

    def add_tokens(self, token_ids: List[int]):
        """Add BOS + tokens + EOS, drain into sequences."""
        doc = [BOS_ID] + token_ids + [EOS_ID]
        self._buffer.extend(doc)
        self._n_tokens += len(doc)
        self._n_docs   += 1
        self._drain()

    def _drain(self):
        while len(self._buffer) >= self.seq_len:
            seq = np.array(self._buffer[:self.seq_len], dtype=DTYPE)
            self._buffer = self._buffer[self.seq_len:]
            self._shard_seqs.append(seq)
            self._n_seqs += 1
            if len(self._shard_seqs) >= self.shard_size:
                self._flush_shard()

    def _flush_shard(self):
        if not self._shard_seqs:
            return
        arr = np.stack(self._shard_seqs, axis=0)
        path = self.output_dir / f"{self.split}_shard_{self._shard_idx:04d}.npy"
        tmp  = path.with_suffix(".tmp.npy")
        np.save(str(tmp), arr)
        tmp.rename(path)
        logger.info(f"    Shard {self._shard_idx:04d}: {len(arr):,} seqs "
                    f"({arr.nbytes / 1e6:.1f} MB) → {path.name}")
        self._shard_idx += 1
        self._shard_seqs = []

    def finalize(self) -> dict:
        self._drain()
        if self._shard_seqs:
            self._flush_shard()
        return {
            "n_documents":     self._n_docs,
            "n_tokens":        self._n_tokens,
            "train_sequences": self._n_seqs,
            "train_tokens":    self._n_seqs * self.seq_len,
            "seq_len":         self.seq_len,
            "n_shards":        self._shard_idx,
        }


def write_val_split(domain_dir: Path, val_fraction: float = 0.005):
    """Split last portion of last shard into a val shard."""
    train_shards = sorted(domain_dir.glob("train_shard_*.npy"))
    if not train_shards:
        return 0
    last = np.load(str(train_shards[-1]))
    total_seqs = sum(np.load(str(s)).shape[0] for s in train_shards)
    n_val = max(1, int(total_seqs * val_fraction))
    n_val = min(n_val, last.shape[0] // 2)
    if n_val < 1:
        return 0
    val_data   = last[-n_val:]
    train_data = last[:-n_val]
    np.save(str(train_shards[-1]), train_data)
    np.save(str(domain_dir / "val_shard_0000.npy"), val_data)
    logger.info(f"    Val split: {n_val} seqs → val_shard_0000.npy")
    return n_val


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Upload
# ─────────────────────────────────────────────────────────────────────────────

def ensure_hf_repo():
    """Create the HF dataset repo if it doesn't exist."""
    from huggingface_hub import HfApi, login
    if not HF_TOKEN:
        raise ValueError(
            "HuggingFace token is empty or not set. Please provide it via the HF_TOKEN environment "
            "variable, Kaggle Secrets, or --hf_token argument."
        )
    login(token=HF_TOKEN)
    api = HfApi()
    try:
        api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset")
        logger.info(f"HF repo exists: {HF_REPO_ID}")
    except Exception:
        api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", private=False)
        logger.info(f"Created HF repo: {HF_REPO_ID}")
    return api


def upload_domain_to_hf(api, domain: str, domain_dir: Path):
    """Upload a single domain's .npy shards + metadata to HF."""
    logger.info(f"[{domain}] Uploading to HF: {HF_REPO_ID}/{domain}/...")
    t0 = time.perf_counter()
    try:
        api.upload_folder(
            folder_path=str(domain_dir),
            path_in_repo=domain,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"Add {domain} tokenized shards",
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"[{domain}] ✓ Uploaded to HF in {elapsed:.0f}s")
        return True
    except Exception as exc:
        logger.error(f"[{domain}] HF upload failed: {exc}")
        traceback.print_exc()
        return False


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Dataset Streaming
# ─────────────────────────────────────────────────────────────────────────────

def stream_hf_dataset(domain: str, cfg: dict) -> Iterator[str]:
    """Stream text documents from HF up to target_docs."""
    from datasets import load_dataset

    text_key    = cfg["text_key"]
    target_docs = cfg["target_docs"]
    hf_id       = cfg["hf_id"]
    hf_config   = cfg.get("hf_config")
    hf_data_dir = cfg.get("hf_data_dir")
    split       = cfg.get("split", "train")

    if domain == "multilingual":
        yield from _stream_multilingual(cfg)
        return

    kwargs = {"streaming": True, "split": split}
    if hf_config:
        kwargs["name"] = hf_config
    if hf_data_dir:
        kwargs["data_dir"] = hf_data_dir

    logger.info(f"  [{domain}] Streaming: {hf_id} (config={hf_config}, "
                f"data_dir={hf_data_dir}, target={target_docs:,})")

    try:
        ds = load_dataset(hf_id, **kwargs)
    except Exception as exc:
        logger.error(f"  [{domain}] Primary source failed: {exc}")
        ds = _try_fallback(domain, cfg, exc)
        if ds is None:
            return

    count = 0
    for example in ds:
        text = _extract_text(example, text_key, domain)
        if text:
            yield text
            count += 1
            if count >= target_docs:
                break
            if count % 500_000 == 0:
                logger.info(f"  [{domain}] Streamed {count:,}/{target_docs:,} docs")


def _extract_text(example: dict, text_key: str, domain: str) -> Optional[str]:
    """Extract text handling domain-specific quirks."""
    if domain == "stackexchange":
        title = example.get("title", "")
        body  = example.get("body", example.get("title_body", ""))
        if body:
            return f"{title}\n\n{body}" if title else body
        return None
    text = example.get(text_key, "")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _stream_multilingual(cfg: dict) -> Iterator[str]:
    """Interleave mC4 languages."""
    from datasets import load_dataset
    langs       = cfg.get("langs", ["de", "fr", "es", "zh", "ja"])
    target_docs = cfg["target_docs"]
    per_lang    = target_docs // len(langs)
    text_key    = cfg["text_key"]
    total = 0
    for lang in langs:
        logger.info(f"  [multilingual] Loading mC4 lang={lang}, target={per_lang:,}")
        try:
            ds = load_dataset("allenai/c4", lang, streaming=True, split="train")
        except Exception as exc:
            logger.warning(f"  [multilingual] Failed lang={lang}: {exc}")
            continue
        lang_count = 0
        for example in ds:
            text = example.get(text_key, "")
            if isinstance(text, str) and text.strip():
                yield text.strip()
                lang_count += 1
                total += 1
                if lang_count >= per_lang:
                    break
                if total >= target_docs:
                    return
        logger.info(f"  [multilingual] lang={lang}: {lang_count:,} docs")


def _try_fallback(domain: str, cfg: dict, original_exc: Exception):
    """Fallback dataset sources when primary fails."""
    from datasets import load_dataset
    fallbacks = {
        "dolma": [
            {"hf_id": "allenai/dolma", "name": "v1_6-sample",
             "streaming": True, "split": "train"},
        ],
        "thestack": [
            {"hf_id": "bigcode/the-stack-v2-train-smol-ids",
             "streaming": True, "split": "train"},
        ],
        "redpajama_cc": [
            {"hf_id": "togethercomputer/RedPajama-Data-1T-Sample",
             "streaming": True, "split": "train"},
        ],
        "fineweb_edu": [
            {"hf_id": "HuggingFaceFW/fineweb-edu-score-2",
             "name": "CC-MAIN-2024-10", "streaming": True, "split": "train"},
        ],
        "openwebmath": [
            {"hf_id": "open-web-math/open-web-math",
             "streaming": True, "split": "train"},
        ],
    }
    if domain not in fallbacks:
        logger.error(f"  [{domain}] No fallback. Skipping.")
        return None
    for fb in fallbacks[domain]:
        hf_id = fb.pop("hf_id")
        logger.warning(f"  [{domain}] Trying fallback: {hf_id}")
        try:
            return load_dataset(hf_id, **fb)
        except Exception as exc2:
            logger.warning(f"  [{domain}] Fallback {hf_id} failed: {exc2}")
    logger.error(f"  [{domain}] All fallbacks failed. Skipping.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Domain Processing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_domain(
    domain:     str,
    cfg:        dict,
    out_base:   Path,
    seq_len:    int = 2048,
    shard_size: int = 50_000,
    tokenizer_profile: str = "standard",
    hf_api=None,
) -> dict:
    """
    Full pipeline for one domain:
      stream HF → quality filter → dedup → tokenize → pack → shard → upload HF
    """
    domain_dir = out_base / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Resumable: skip if already done
    meta_path = domain_dir / "packing_meta.json"
    if meta_path.exists():
        logger.info(f"[{domain}] Already processed — skipping (delete {meta_path} to rerun)")
        with open(str(meta_path)) as f:
            return json.load(f)

    t0 = time.perf_counter()

    # ── Initialize tokenizer ─────────────────────────────────────────────────
    try:
        from tokenizer.crayon_wrapper import ForgeTokenizer
        tok = ForgeTokenizer(
            profile=tokenizer_profile,
            device="cpu",
            n_workers=1,
            max_length=seq_len * 4,
            add_bos=False,
            add_eos=False,
        )
        logger.info(f"[{domain}] CRAYON tokenizer loaded")
    except ImportError:
        logger.error(f"[{domain}] CRAYON not found! pip install xerv-crayon")
        return {"domain": domain, "error": "CRAYON tokenizer not found"}

    # ── Initialize packer + filters ──────────────────────────────────────────
    packer = InlinePacker(output_dir=domain_dir, seq_len=seq_len,
                          shard_size=shard_size)
    use_quality = cfg.get("quality_filter", True)
    dedup       = DedupFilter()

    n_streamed = n_quality_fail = n_dup = n_tokenized = n_tok_fail = 0

    # ── Stream + filter + tokenize + pack ────────────────────────────────────
    logger.info(f"[{domain}] Starting (target={cfg['target_docs']:,} docs, "
                f"{cfg['target_tokens']/1e9:.0f}B tokens, {cfg['weight_pct']}% of mix)...")

    try:
        for text in stream_hf_dataset(domain, cfg):
            n_streamed += 1

            if use_quality and not is_quality_document(text):
                n_quality_fail += 1
                continue
            if dedup.is_duplicate(text):
                n_dup += 1
                continue

            try:
                token_ids = tok.encode(text, add_bos=False, add_eos=False,
                                       truncate=False)
                if not token_ids:
                    continue
            except Exception:
                n_tok_fail += 1
                continue

            packer.add_tokens(token_ids)
            n_tokenized += 1

            if n_tokenized % 200_000 == 0:
                elapsed = time.perf_counter() - t0
                rate = n_tokenized / elapsed
                est_remain = (cfg["target_docs"] - n_streamed) / max(rate, 1)
                tok_so_far = packer._n_seqs * seq_len
                logger.info(
                    f"[{domain}] {n_tokenized:,} tokenized | "
                    f"{tok_so_far/1e9:.2f}B tokens packed | "
                    f"{n_quality_fail:,} qf | {n_dup:,} dup | "
                    f"{elapsed:.0f}s | ETA {est_remain:.0f}s"
                )

    except Exception as exc:
        logger.error(f"[{domain}] Stream error after {n_streamed:,} docs: {exc}")
        traceback.print_exc()

    # ── Finalize ─────────────────────────────────────────────────────────────
    pack_meta = packer.finalize()
    n_val = write_val_split(domain_dir, val_fraction=0.005)

    elapsed = time.perf_counter() - t0
    train_seqs = pack_meta["train_sequences"] - n_val
    meta = {
        "domain":          domain,
        "hf_source":       cfg["hf_id"],
        "weight_pct":      cfg["weight_pct"],
        "n_streamed":      n_streamed,
        "n_quality_fail":  n_quality_fail,
        "n_dup":           n_dup,
        "n_tokenized":     n_tokenized,
        "n_tok_fail":      n_tok_fail,
        "train_sequences": train_seqs,
        "val_sequences":   n_val,
        "train_tokens":    train_seqs * seq_len,
        "val_tokens":      n_val * seq_len,
        "seq_len":         seq_len,
        "n_shards":        pack_meta["n_shards"],
        "elapsed_s":       round(elapsed, 1),
    }

    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"[{domain}] ✓ DONE: {n_tokenized:,} docs → "
        f"{train_seqs:,} train seqs ({meta['train_tokens']/1e9:.2f}B tokens) | "
        f"{elapsed/60:.1f}min"
    )

    # ── Upload to HuggingFace ────────────────────────────────────────────────
    if hf_api is not None:
        upload_domain_to_hf(hf_api, domain, domain_dir)

        # Free disk space after upload (Kaggle has ~70GB limit)
        # Keep only the metadata, remove heavy .npy files
        for npy_file in domain_dir.glob("*.npy"):
            npy_file.unlink()
        logger.info(f"[{domain}] Freed disk space (shards uploaded to HF)")

    # Free memory
    del tok, packer, dedup
    gc.collect()

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    out_base:   str = OUT_DIR,
    seq_len:    int = 2048,
    shard_size: int = 50_000,
    tokenizer_profile: str = "standard",
    domains: Optional[List[str]] = None,
    skip_domains: Optional[List[str]] = None,
    upload: bool = True,
    hf_username: Optional[str] = None,
    hf_token: Optional[str] = None,
):
    """Run the full 10-domain pipeline, upload each domain to HF after processing."""
    global HF_USERNAME, HF_TOKEN, HF_REPO_ID
    if hf_username:
        HF_USERNAME = hf_username
    if hf_token:
        HF_TOKEN = hf_token
    HF_REPO_ID = f"{HF_USERNAME}/forge-3b-pretrain-data"

    out_path = Path(out_base)
    out_path.mkdir(parents=True, exist_ok=True)

    target_domains = domains or list(DATASET_REGISTRY.keys())
    if skip_domains:
        target_domains = [d for d in target_domains if d not in skip_domains]

    # ── Setup HF ─────────────────────────────────────────────────────────────
    hf_api = None
    if upload:
        try:
            hf_api = ensure_hf_repo()
        except Exception as exc:
            logger.error(f"HF setup failed: {exc}. Continuing without upload.")
            hf_api = None

    # ── Banner ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'═'*70}")
    logger.info(f"  FORGE-3B Data Pipeline — Production Run")
    logger.info(f"  Output:   {out_path}")
    logger.info(f"  HF Repo:  {HF_REPO_ID}")
    logger.info(f"  Domains:  {target_domains}")
    logger.info(f"  Seq len:  {seq_len}")
    logger.info(f"  Target:   ≥{TOTAL_TARGET_TOKENS/1e9:.0f}B total tokens")
    logger.info(f"{'═'*70}\n")

    t0_global = time.perf_counter()
    summaries = []

    # ── Process each domain sequentially ─────────────────────────────────────
    # Sequential is more reliable on Kaggle (avoids memory pressure).
    # Each domain is uploaded + disk-freed before starting the next.
    for i, domain in enumerate(target_domains, 1):
        cfg = DATASET_REGISTRY[domain]
        logger.info(f"\n{'─'*60}")
        logger.info(f" [{i}/{len(target_domains)}] Domain: {domain.upper()}")
        logger.info(f" {cfg.get('weight_pct', '?')}% of mix | "
                    f"Target: {cfg['target_tokens']/1e9:.0f}B tokens")
        logger.info(f"{'─'*60}")

        try:
            meta = process_domain(
                domain=domain,
                cfg=cfg,
                out_base=out_path,
                seq_len=seq_len,
                shard_size=shard_size,
                tokenizer_profile=tokenizer_profile,
                hf_api=hf_api,
            )
            summaries.append(meta)
        except Exception as exc:
            logger.error(f"[{domain}] FATAL: {exc}")
            traceback.print_exc()
            summaries.append({"domain": domain, "error": str(exc)})

    # ── Upload global manifest ───────────────────────────────────────────────
    manifest_path = out_path / "preprocessing_manifest.json"
    with open(str(manifest_path), "w") as f:
        json.dump(summaries, f, indent=2)

    if hf_api is not None:
        try:
            hf_api.upload_file(
                path_or_fileobj=str(manifest_path),
                path_in_repo="preprocessing_manifest.json",
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message="Add preprocessing manifest",
            )
            logger.info("✓ Uploaded manifest to HF")
        except Exception as exc:
            logger.warning(f"Manifest upload failed: {exc}")

    # ── Create dataset card on HF ────────────────────────────────────────────
    if hf_api is not None:
        _upload_dataset_card(hf_api, summaries)

    elapsed_global = time.perf_counter() - t0_global

    # ── Print summary table ──────────────────────────────────────────────────
    print(f"\n{'═'*85}")
    print(f"  FORGE-3B Data Pipeline — COMPLETE")
    print(f"  Total time: {elapsed_global/3600:.1f} hours ({elapsed_global:.0f}s)")
    print(f"  HF Repo: https://huggingface.co/datasets/{HF_REPO_ID}")
    print(f"{'═'*85}")
    print(f"{'Domain':<18} {'Train Seqs':>12} {'Val Seqs':>10} "
          f"{'Tokens (B)':>12} {'Time (min)':>12} {'Status':>8}")
    print(f"{'─'*85}")

    total_train_tokens = 0
    total_val_tokens   = 0

    for m in summaries:
        if "error" in m:
            print(f"{m.get('domain','?'):<18}  {'—':>12} {'—':>10} "
                  f"{'—':>12} {'—':>12}     ✗")
            print(f"   Error: {m['error'][:55]}")
        else:
            tt = m.get("train_tokens", 0)
            vt = m.get("val_tokens", 0)
            total_train_tokens += tt
            total_val_tokens   += vt
            print(
                f"{m.get('domain','?'):<18} "
                f"{m.get('train_sequences',0):>12,} "
                f"{m.get('val_sequences',0):>10,} "
                f"{tt/1e9:>12.2f} "
                f"{m.get('elapsed_s',0)/60:>12.1f} "
                f"     ✓"
            )

    total_all = total_train_tokens + total_val_tokens
    print(f"{'─'*85}")
    print(f"{'TOTAL':<18} {'':>12} {'':>10} "
          f"{total_all/1e9:>12.2f} "
          f"{elapsed_global/60:>12.1f}")
    print(f"{'═'*85}")

    # ── Token budget verification ────────────────────────────────────────────
    if total_all >= TOTAL_TARGET_TOKENS:
        print(f"\n✅ TOKEN BUDGET MET: {total_all/1e9:.2f}B ≥ {TOTAL_TARGET_TOKENS/1e9:.0f}B")
    else:
        deficit = TOTAL_TARGET_TOKENS - total_all
        print(f"\n⚠️  TOKEN BUDGET SHORT: {total_all/1e9:.2f}B < "
              f"{TOTAL_TARGET_TOKENS/1e9:.0f}B (deficit: {deficit/1e9:.2f}B)")
        print("   Re-run with higher --target_docs or check failed domains.")

    print(f"\n📦 All data uploaded to: https://huggingface.co/datasets/{HF_REPO_ID}")
    print(f"🚀 Ready for training: python run_pretrain.py --data_dir <hf_download_path>")


def _upload_dataset_card(api, summaries: list):
    """Create a README.md dataset card on HF."""
    total_tokens = sum(
        m.get("train_tokens", 0) + m.get("val_tokens", 0)
        for m in summaries if "error" not in m
    )
    n_domains = sum(1 for m in summaries if "error" not in m)

    card = f"""---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
  - de
  - fr
  - es
  - zh
  - ja
  - ru
  - pt
  - it
  - nl
  - ar
tags:
  - forge-3b
  - pretraining
  - tokenized
  - packed
size_categories:
  - 10B<n<100B
---

# FORGE-3B Pretraining Data

Tokenized and packed pretraining data for the FORGE-3B language model.

## Stats
- **Total tokens**: {total_tokens/1e9:.2f}B
- **Domains**: {n_domains}/10
- **Sequence length**: 2048 tokens
- **Format**: `.npy` shards of shape `(N, 2048)` with dtype `uint32`
- **Tokenizer**: CRAYON (xerv-crayon, standard profile)

## Domain Breakdown

| Domain | Weight | Tokens (B) | Status |
|:-------|-------:|-----------:|:------:|
"""
    for m in summaries:
        d = m.get("domain", "?")
        if "error" in m:
            card += f"| {d} | — | — | ✗ |\n"
        else:
            tt = (m.get("train_tokens", 0) + m.get("val_tokens", 0)) / 1e9
            card += f"| {d} | {m.get('weight_pct', '?')}% | {tt:.2f} | ✓ |\n"

    card += f"""
## Usage

```python
import numpy as np
from huggingface_hub import hf_hub_download

# Download a shard
path = hf_hub_download(
    repo_id="{HF_REPO_ID}",
    filename="fineweb_edu/train_shard_0000.npy",
    repo_type="dataset",
)
data = np.load(path)  # shape: (50000, 2048), dtype: uint32
```

## Structure
```
{HF_REPO_ID}/
├── fineweb_edu/     (30%, 15B tokens)
├── thestack/        (16%, 8B tokens)
├── wikipedia/       (8%, 4B tokens)
├── openwebmath/     (8%, 4B tokens)
├── books/           (7%, 3.5B tokens)
├── arxiv/           (6%, 3B tokens)
├── dolma/           (10%, 5B tokens)
├── stackexchange/   (5%, 2.5B tokens)
├── redpajama_cc/    (6%, 3B tokens)
├── multilingual/    (4%, 2B tokens)
└── preprocessing_manifest.json
```
"""
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(card)
            card_path = f.name
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        os.unlink(card_path)
        logger.info("✓ Dataset card uploaded to HF")
    except Exception as exc:
        logger.warning(f"Dataset card upload failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FORGE-3B Data Pipeline — Stream, tokenize, pack, upload to HF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help="Local output directory for packed shards")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length for packing")
    parser.add_argument("--shard_size", type=int, default=50_000,
                        help="Sequences per .npy shard")
    parser.add_argument("--tokenizer_profile", default="standard",
                        choices=["standard", "lite"])
    parser.add_argument("--domains", nargs="*", default=None,
                        help="Only these domains (e.g. arxiv books)")
    parser.add_argument("--skip_domains", nargs="*", default=None,
                        help="Skip these domains")
    parser.add_argument("--no_upload", action="store_true",
                        help="Skip HuggingFace upload")
    parser.add_argument("--hf_username", default=None,
                        help="HuggingFace username / organization")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace user access token")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_pipeline(
        out_base=args.out_dir,
        seq_len=args.seq_len,
        shard_size=args.shard_size,
        tokenizer_profile=args.tokenizer_profile,
        domains=args.domains,
        skip_domains=args.skip_domains,
        upload=not args.no_upload,
        hf_username=args.hf_username,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
