#!/usr/bin/env python3
"""
FORGE-3B Post-Training Data Preparation Script.
Downloads and preps standard SFT and DPO datasets from Hugging Face Hub,
converts them to the JSONL formats expected by run_sft.py and run_dpo.py,
and saves them to the data directory.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prep_post_training_data")


def parse_args():
    parser = argparse.ArgumentParser(description="FORGE-3B SFT & DPO Dataset Preparation")
    
    # Paths
    parser.add_argument("--sft_out_dir", type=str, default="./data/sft",
                        help="Output directory for SFT data")
    parser.add_argument("--dpo_out_dir", type=str, default="./data/dpo",
                        help="Output directory for DPO data")
    
    # Limits for testing/debugging
    parser.add_argument("--max_sft_samples", type=int, default=150000,
                        help="Maximum SFT samples to download & prepare (~1.4B tokens target)")
    parser.add_argument("--max_dpo_samples", type=int, default=50000,
                        help="Maximum DPO preference pairs to download & prepare")
    parser.add_argument("--quick_run", action="store_true",
                        help="If set, prepare only a tiny subset (100 samples) for testing")
    
    return parser.parse_args()


def download_and_format_sft(out_dir: str, max_samples: int):
    """
    Download and combine SFT datasets:
    1. HuggingFaceH4/ultrachat_200k (conversational)
    2. Open-Orca/OpenOrca (instruction/reasoning)
    3. meta-math/MetaMathQA (math QA)
    """
    from datasets import load_dataset
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(out_dir) / "train.jsonl"
    
    logger.info("=" * 60)
    logger.info(f"PREPARING SFT DATASET -> {out_file}")
    logger.info("=" * 60)
    
    sft_data = []
    
    # ── 1. UltraChat-200k ─────────────────────────────────────────────────────
    logger.info("Loading HuggingFaceH4/ultrachat_200k...")
    try:
        ds_uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", trust_remote_code=True)
        # Shuffle and limit
        ds_uc = ds_uc.shuffle(seed=42)
        uc_samples = min(len(ds_uc), max_samples // 3)
        
        logger.info(f"Formatting {uc_samples:,} samples from UltraChat...")
        for i in tqdm(range(uc_samples), desc="UltraChat"):
            item = ds_uc[i]
            # Already formatted in HF chat template format under 'messages'
            sft_data.append({"messages": item["messages"]})
    except Exception as e:
        logger.error(f"Failed to load/process UltraChat-200k: {e}")
        
    # ── 2. Open-Orca ─────────────────────────────────────────────────────────
    logger.info("Loading Open-Orca/OpenOrca...")
    try:
        ds_oo = load_dataset("Open-Orca/OpenOrca", split="train", trust_remote_code=True)
        ds_oo = ds_oo.shuffle(seed=42)
        oo_samples = min(len(ds_oo), max_samples // 3)
        
        logger.info(f"Formatting {oo_samples:,} samples from Open-Orca...")
        for i in tqdm(range(oo_samples), desc="Open-Orca"):
            item = ds_oo[i]
            # Convert system_prompt, question, response to standard messages format
            messages = []
            if item.get("system_prompt"):
                messages.append({"role": "system", "content": item["system_prompt"]})
            messages.append({"role": "user", "content": item["question"]})
            messages.append({"role": "assistant", "content": item["response"]})
            sft_data.append({"messages": messages})
    except Exception as e:
        logger.error(f"Failed to load/process Open-Orca: {e}")

    # ── 3. MetaMathQA ──────────────────────────────────────────────────────────
    logger.info("Loading meta-math/MetaMathQA...")
    try:
        ds_mm = load_dataset("meta-math/MetaMathQA", split="train", trust_remote_code=True)
        ds_mm = ds_mm.shuffle(seed=42)
        mm_samples = min(len(ds_mm), max_samples // 3)
        
        logger.info(f"Formatting {mm_samples:,} samples from MetaMathQA...")
        for i in tqdm(range(mm_samples), desc="MetaMathQA"):
            item = ds_mm[i]
            messages = [
                {"role": "user", "content": item["query"]},
                {"role": "assistant", "content": item["response"]}
            ]
            sft_data.append({"messages": messages})
    except Exception as e:
        logger.error(f"Failed to load/process MetaMathQA: {e}")

    # ── Write output ──────────────────────────────────────────────────────────
    if not sft_data:
        logger.error("No SFT data collected. Skipping SFT dataset generation.")
        return
        
    logger.info(f"Writing {len(sft_data):,} total SFT samples to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info(f"SFT dataset preparation completed successfully! File size: {out_file.stat().st_size / 1e6:.2f} MB")


def download_and_format_dpo(out_dir: str, max_samples: int):
    """
    Download and combine DPO preference datasets:
    1. argilla/ultrafeedback-binarized-preferences-cleaned (highly clean alignment dataset)
    2. Anthropic/hh-rlhf (safety alignment)
    """
    from datasets import load_dataset
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(out_dir) / "preferences.jsonl"
    
    logger.info("=" * 60)
    logger.info(f"PREPARING DPO DATASET -> {out_file}")
    logger.info("=" * 60)
    
    dpo_data = []
    
    # ── 1. UltraFeedback Binarized ────────────────────────────────────────────
    logger.info("Loading argilla/ultrafeedback-binarized-preferences-cleaned...")
    try:
        ds_uf = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train", trust_remote_code=True)
        ds_uf = ds_uf.shuffle(seed=42)
        uf_samples = min(len(ds_uf), max_samples // 2)
        
        logger.info(f"Formatting {uf_samples:,} pairs from UltraFeedback...")
        for i in tqdm(range(uf_samples), desc="UltraFeedback"):
            item = ds_uf[i]
            # Convert chosen/rejected message arrays to DPO format
            # Format expected: prompt, chosen string, rejected string
            prompt = item["prompt"]
            prompt_msgs = [{"role": "user", "content": prompt}]
            
            # Extract chosen and rejected text
            chosen_text = ""
            for m in item["chosen"]:
                if m["role"] == "assistant":
                    chosen_text = m["content"]
            
            rejected_text = ""
            for m in item["rejected"]:
                if m["role"] == "assistant":
                    rejected_text = m["content"]
            
            if chosen_text and rejected_text:
                dpo_data.append({
                    "prompt": prompt_msgs,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                })
    except Exception as e:
        logger.error(f"Failed to load/process UltraFeedback: {e}")

    # ── 2. Anthropic HH-RLHF ──────────────────────────────────────────────────
    logger.info("Loading Anthropic/hh-rlhf...")
    try:
        ds_hh = load_dataset("Anthropic/hh-rlhf", split="train", trust_remote_code=True)
        ds_hh = ds_hh.shuffle(seed=42)
        hh_samples = min(len(ds_hh), max_samples // 2)
        
        def parse_hh_dialog(dialog_str: str):
            """Parse Anthropic conversational text into structured role messages."""
            turns = dialog_str.split("\n\n")
            messages = []
            for turn in turns:
                turn = turn.strip()
                if not turn:
                    continue
                if turn.startswith("Human:"):
                    messages.append({"role": "user", "content": turn[len("Human:"):].strip()})
                elif turn.startswith("Assistant:"):
                    messages.append({"role": "assistant", "content": turn[len("Assistant:"):].strip()})
            return messages

        logger.info(f"Formatting {hh_samples:,} pairs from HH-RLHF...")
        for i in tqdm(range(hh_samples), desc="HH-RLHF"):
            item = ds_hh[i]
            chosen_turns = parse_hh_dialog(item["chosen"])
            rejected_turns = parse_hh_dialog(item["rejected"])
            
            # Check prefix alignment and find prompt vs response splits
            if not chosen_turns or not rejected_turns:
                continue
                
            # The prompt turns are identical in chosen/rejected up to the last turn
            prompt_turns = []
            for ct, rt in zip(chosen_turns, rejected_turns):
                if ct == rt:
                    prompt_turns.append(ct)
                else:
                    break
            
            # The final response differs
            chosen_text = chosen_turns[len(prompt_turns)]["content"] if len(chosen_turns) > len(prompt_turns) else ""
            rejected_text = rejected_turns[len(prompt_turns)]["content"] if len(rejected_turns) > len(prompt_turns) else ""
            
            if prompt_turns and chosen_text and rejected_text:
                dpo_data.append({
                    "prompt": prompt_turns,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                })
    except Exception as e:
        logger.error(f"Failed to load/process Anthropic HH-RLHF: {e}")

    # ── Write output ──────────────────────────────────────────────────────────
    if not dpo_data:
        logger.error("No DPO data collected. Skipping DPO dataset generation.")
        return
        
    logger.info(f"Writing {len(dpo_data):,} total DPO preference pairs to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logger.info(f"DPO dataset preparation completed successfully! File size: {out_file.stat().st_size / 1e6:.2f} MB")


def main():
    args = parse_args()
    
    # Check dependencies
    try:
        import datasets
    except ImportError:
        logger.error("Hugging Face 'datasets' library is missing! Install it via: pip install datasets tqdm")
        sys.exit(1)
        
    sft_samples = args.max_sft_samples
    dpo_samples = args.max_dpo_samples
    
    if args.quick_run:
        sft_samples = 150
        dpo_samples = 100
        logger.info(f"⚡ Running QUICK_RUN: limit to {sft_samples} SFT samples and {dpo_samples} DPO pairs.")
    
    # 1. SFT Prep
    download_and_format_sft(args.sft_out_dir, sft_samples)
    
    # 2. DPO Prep
    download_and_format_dpo(args.dpo_out_dir, dpo_samples)
    
    logger.info("=" * 60)
    logger.info("POST-TRAINING DATASET PREPARATION COMPLETED!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
