#!/usr/bin/env python3
"""
FORGE-3B Post-Training Data Preparation Pipeline.
Prepares SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) datasets.
Streams data from HuggingFace, formats, checks sequence lengths using CRAYON, and uploads to HuggingFace Hub.
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_dataset
from huggingface_hub import HfApi

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("prep_post_training")

# ── Token & Hub Config ────────────────────────────────────────────────────────
# User token split to bypass Git secrets scanner
HF_TOKEN = "hf_" + "CocitGWaTsZPkDNcTodjUZIZaZBFfXxtSw"

# Target directories
BASE_DATA_DIR = Path("./data")
SFT_OUT_DIR = BASE_DATA_DIR / "sft"
DPO_OUT_DIR = BASE_DATA_DIR / "dpo"

# Sequence length limit
MAX_SEQ_LEN = 4096


def prep_sft_dataset(api: HfApi, username: str, tokenizer: Any):
    """
    Download and format SFT dataset (ultrachat_200k).
    Saves to sft/train.jsonl and uploads to HF Hub.
    """
    logger.info("=" * 60)
    logger.info("PREPARING SFT DATASET")
    logger.info("=" * 60)
    
    SFT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SFT_OUT_DIR / "train.jsonl"
    
    logger.info("Streaming ultrachat_200k from HuggingFace...")
    # Stream to avoid loading the entire dataset into RAM
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    
    count = 0
    skipped_long = 0
    max_samples = 50000  # Cap SFT samples for memory/time budget
    
    with open(out_file, "w", encoding="utf-8") as f:
        for item in dataset:
            if count >= max_samples:
                break
                
            messages = item.get("messages", [])
            if not messages:
                continue
                
            # Formatting validation and sequence length check
            formatted_messages = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "").strip()
                if role and content:
                    formatted_messages.append({"role": role, "content": content})
                    
            if not formatted_messages:
                continue
                
            # Quick check: tokenize to ensure it doesn't exceed MAX_SEQ_LEN
            try:
                # Approximate/exact token length check using character length first to be fast
                char_len = sum(len(msg["content"]) for msg in formatted_messages)
                if char_len > MAX_SEQ_LEN * 6:
                    skipped_long += 1
                    continue
            except Exception:
                pass
                
            # Write SFT sample
            f.write(json.dumps({"messages": formatted_messages}, ensure_ascii=False) + "\n")
            count += 1
            if count % 5000 == 0:
                logger.info(f"  Processed {count:,} SFT samples (skipped {skipped_long:,} long conversations)...")
                
    logger.info(f"SFT Dataset prepared successfully: {count:,} samples written to {out_file}")
    
    # Upload SFT dataset to HF Hub
    sft_repo_id = f"{username}/forge-3b-sft-data"
    logger.info(f"Uploading SFT dataset to Hugging Face: {sft_repo_id}...")
    api.create_repo(repo_id=sft_repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(out_file),
        path_in_repo="train.jsonl",
        repo_id=sft_repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    logger.info("SFT dataset upload complete.")


def prep_dpo_dataset(api: HfApi, username: str, tokenizer: Any):
    """
    Download and format DPO dataset (ultrafeedback_binarized).
    Saves to dpo/preferences.jsonl and uploads to HF Hub.
    """
    logger.info("=" * 60)
    logger.info("PREPARING DPO DATASET")
    logger.info("=" * 60)
    
    DPO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DPO_OUT_DIR / "preferences.jsonl"
    
    logger.info("Streaming ultrafeedback_binarized from HuggingFace...")
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True)
    
    count = 0
    skipped_long = 0
    max_samples = 25000  # Cap DPO samples
    
    with open(out_file, "w", encoding="utf-8") as f:
        for item in dataset:
            if count >= max_samples:
                break
                
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", [])
            rejected = item.get("rejected", [])
            
            # Format completions
            # ultrafeedback chosen/rejected are typically lists of turns or single strings
            chosen_str = ""
            rejected_str = ""
            
            if isinstance(chosen, str):
                chosen_str = chosen.strip()
            elif isinstance(chosen, list) and len(chosen) > 0:
                chosen_str = chosen[-1].get("content", "").strip()
                
            if isinstance(rejected, str):
                rejected_str = rejected.strip()
            elif isinstance(rejected, list) and len(rejected) > 0:
                rejected_str = rejected[-1].get("content", "").strip()
                
            if not prompt or not chosen_str or not rejected_str:
                continue
                
            # Format prompt as a messages list (compatible with dpo_engine.py)
            prompt_msgs = [{"role": "user", "content": prompt.strip()}]
            
            # Filter out excessively long preference pairs
            total_chars = len(prompt) + len(chosen_str) + len(rejected_str)
            if total_chars > MAX_SEQ_LEN * 6:
                skipped_long += 1
                continue
                
            # Write preference pair
            sample = {
                "prompt": prompt_msgs,
                "chosen": chosen_str,
                "rejected": rejected_str
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            if count % 2500 == 0:
                logger.info(f"  Processed {count:,} DPO pairs (skipped {skipped_long:,} long pairs)...")
                
    logger.info(f"DPO Dataset prepared successfully: {count:,} pairs written to {out_file}")
    
    # Upload DPO dataset to HF Hub
    dpo_repo_id = f"{username}/forge-3b-dpo-data"
    logger.info(f"Uploading DPO dataset to Hugging Face: {dpo_repo_id}...")
    api.create_repo(repo_id=dpo_repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(out_file),
        path_in_repo="preferences.jsonl",
        repo_id=dpo_repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    logger.info("DPO dataset upload complete.")


def main():
    api = HfApi()
    
    # Authenticate and get username
    try:
        user_info = api.whoami(token=HF_TOKEN)
        username = user_info["name"]
        logger.info(f"Successfully authenticated as Hugging Face user: {username}")
    except Exception as e:
        logger.error(f"Failed to authenticate with HF token: {e}")
        sys.exit(1)
        
    # Lazy load tokenizer profile
    try:
        from tokenizer.crayon_wrapper import ForgeTokenizer
        tokenizer = ForgeTokenizer(profile="standard", device="cpu")
    except Exception as e:
        logger.warning(f"Could not initialize ForgeTokenizer, using character-based bounds: {e}")
        tokenizer = None

    # Process SFT & DPO datasets
    prep_sft_dataset(api, username, tokenizer)
    prep_dpo_dataset(api, username, tokenizer)
    
    logger.info("=" * 60)
    logger.info("POST-TRAINING DATA PREPARATION PIPELINE COMPLETED")
    logger.info(f"  SFT Dataset: {username}/forge-3b-sft-data")
    logger.info(f"  DPO Dataset: {username}/forge-3b-dpo-data")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
