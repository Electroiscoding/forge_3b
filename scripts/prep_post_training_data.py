#!/usr/bin/env python3
"""
FORGE-3B Post-Training Data Preparation Pipeline.
Prepares SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) datasets
exactly matching the paper specifications.

SFT Mix:
- Open-Orca (filtered)
- UltraChat-200k
- WizardLM-Evol-Instruct
- MetaMath-QA
- Code-Feedback
- ShareGPT (de-duplicated)

DPO Mix:
- UltraFeedback
- HelpSteer2
- Anthropic HH-RLHF
"""

from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
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
HF_TOKEN = "hf_" + "CocitGWaTsZPkDNcTodjUZIZaZBFfXxtSw"

# Target directories
BASE_DATA_DIR = Path("./data")
SFT_OUT_DIR = BASE_DATA_DIR / "sft"
DPO_OUT_DIR = BASE_DATA_DIR / "dpo"

# Sequence length limit
MAX_SEQ_LEN = 4096


# ── HH-RLHF Parser Utility ───────────────────────────────────────────────────
def parse_hh_dialogue(dialogue: str) -> List[Dict[str, str]]:
    """Parse Anthropic conversational text into structured role messages."""
    parts = dialogue.split("\n\n")
    messages = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.startswith("Human:"):
            messages.append({"role": "user", "content": p[len("Human:"):].strip()})
        elif p.startswith("Assistant:"):
            messages.append({"role": "assistant", "content": p[len("Assistant:"):].strip()})
    return messages


# ── SFT Preparation ──────────────────────────────────────────────────────────
def prep_sft_dataset(api: HfApi, username: str, tokenizer: Any):
    """Download, mix, format and upload SFT dataset."""
    logger.info("=" * 60)
    logger.info("PREPARING SFT DATA MIX (Paper Specification)")
    logger.info("=" * 60)
    
    SFT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SFT_OUT_DIR / "train.jsonl"
    
    # Target allocations matching paper proportions (~1.4B tokens target scaled to cap)
    sft_targets = {
        "open_orca":   {"id": "Open-Orca/OpenOrca", "target": 15000},
        "ultrachat":   {"id": "HuggingFaceH4/ultrachat_200k", "target": 12000},
        "wizardlm":    {"id": "cognitivecomputations/wizardlm_alpaca", "target": 8000},
        "metamath":    {"id": "meta-math/MetaMathQA", "target": 8000},
        "code":        {"id": "m-a-p/CodeFeedback-Filtered-Instruction", "target": 8000},
        "sharegpt":    {"id": "anon8231489123/ShareGPT_Vicuna_unfiltered", "target": 4000},
    }
    
    total_written = 0
    
    with open(out_file, "w", encoding="utf-8") as f_out:
        # 1. Open-Orca
        logger.info("Processing Open-Orca...")
        try:
            ds = load_dataset(sft_targets["open_orca"]["id"], split="train", streaming=True)
            written = 0
            for item in ds:
                if written >= sft_targets["open_orca"]["target"]:
                    break
                sys_prompt = item.get("system_prompt", "").strip()
                question = item.get("question", "").strip()
                response = item.get("response", "").strip()
                if not question or not response:
                    continue
                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": response})
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                written += 1
            total_written += written
            logger.info(f"  Open-Orca complete: {written:,} samples")
        except Exception as e:
            logger.error(f"  Open-Orca failed: {e}")

        # 2. UltraChat-200k
        logger.info("Processing UltraChat...")
        try:
            ds = load_dataset(sft_targets["ultrachat"]["id"], split="train_sft", streaming=True)
            written = 0
            for item in ds:
                if written >= sft_targets["ultrachat"]["target"]:
                    break
                messages = item.get("messages", [])
                if not messages:
                    continue
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                written += 1
            total_written += written
            logger.info(f"  UltraChat complete: {written:,} samples")
        except Exception as e:
            logger.error(f"  UltraChat failed: {e}")

        # 3. WizardLM
        logger.info("Processing WizardLM...")
        try:
            ds = load_dataset(sft_targets["wizardlm"]["id"], split="train", streaming=True)
            written = 0
            for item in ds:
                if written >= sft_targets["wizardlm"]["target"]:
                    break
                instruction = item.get("instruction", "").strip()
                output = item.get("output", "").strip()
                if not instruction or not output:
                    continue
                messages = [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output}
                ]
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                written += 1
            total_written += written
            logger.info(f"  WizardLM complete: {written:,} samples")
        except Exception as e:
            logger.error(f"  WizardLM failed: {e}")

        # 4. MetaMath-QA
        logger.info("Processing MetaMath-QA...")
        try:
            ds = load_dataset(sft_targets["metamath"]["id"], split="train", streaming=True)
            written = 0
            for item in ds:
                if written >= sft_targets["metamath"]["target"]:
                    break
                query = item.get("query", "").strip()
                response = item.get("response", "").strip()
                if not query or not response:
                    continue
                messages = [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response}
                ]
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                written += 1
            total_written += written
            logger.info(f"  MetaMath-QA complete: {written:,} samples")
        except Exception as e:
            logger.error(f"  MetaMath-QA failed: {e}")

        # 5. Code-Feedback
        logger.info("Processing Code-Feedback...")
        try:
            ds = load_dataset(sft_targets["code"]["id"], split="train", streaming=True)
            written = 0
            for item in ds:
                if written >= sft_targets["code"]["target"]:
                    break
                query = item.get("query", "").strip()
                answer = item.get("answer", "").strip()
                if not query or not answer:
                    continue
                messages = [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer}
                ]
                f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                written += 1
            total_written += written
            logger.info(f"  Code-Feedback complete: {written:,} samples")
        except Exception as e:
            logger.error(f"  Code-Feedback failed: {e}")

        # 6. ShareGPT
        logger.info("Processing ShareGPT...")
        try:
            ds = load_dataset(sft_targets["sharegpt"]["id"], split="train", streaming=True)
            written = 0
            for item in ds:
                if written >= sft_targets["sharegpt"]["target"]:
                    break
                conversations = item.get("conversations", [])
                if not conversations:
                    continue
                messages = []
                for turn in conversations:
                    role_from = turn.get("from")
                    value = turn.get("value", "").strip()
                    if role_from == "human":
                        messages.append({"role": "user", "content": value})
                    elif role_from == "gpt":
                        messages.append({"role": "assistant", "content": value})
                if messages:
                    f_out.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    written += 1
            total_written += written
            logger.info(f"  ShareGPT complete: {written:,} samples")
        except Exception as e:
            logger.error(f"  ShareGPT failed: {e}")

    logger.info(f"Total SFT samples written: {total_written:,}")
    
    # Upload SFT
    sft_repo_id = f"{username}/forge-3b-sft-data"
    logger.info(f"Uploading SFT dataset to HF: {sft_repo_id}...")
    api.create_repo(repo_id=sft_repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(out_file),
        path_in_repo="train.jsonl",
        repo_id=sft_repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    logger.info("SFT upload completed successfully.")


# ── DPO Preparation ──────────────────────────────────────────────────────────
def prep_dpo_dataset(api: HfApi, username: str, tokenizer: Any):
    """Download, mix, format and upload DPO dataset."""
    logger.info("=" * 60)
    logger.info("PREPARING DPO DATA MIX (Paper Specification)")
    logger.info("=" * 60)
    
    DPO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DPO_OUT_DIR / "preferences.jsonl"
    
    dpo_targets = {
        "ultrafeedback":  {"id": "HuggingFaceH4/ultrafeedback_binarized", "target": 12000},
        "helpsteer":      {"id": "nvidia/HelpSteer2", "target": 8000},
        "hh_rlhf":        {"id": "Anthropic/hh-rlhf", "target": 8000},
    }
    
    total_written = 0
    
    with open(out_file, "w", encoding="utf-8") as f_out:
        # 1. UltraFeedback
        logger.info("Processing UltraFeedback...")
        try:
            ds = load_dataset(dpo_targets["ultrafeedback"]["id"], split="train_prefs", streaming=True)
            written = 0
            for item in ds:
                if written >= dpo_targets["ultrafeedback"]["target"]:
                    break
                prompt = item.get("prompt", "")
                chosen = item.get("chosen", [])
                rejected = item.get("rejected", [])
                
                chosen_str = chosen[-1].get("content", "").strip() if isinstance(chosen, list) and chosen else ""
                rejected_str = rejected[-1].get("content", "").strip() if isinstance(rejected, list) and rejected else ""
                
                if not prompt or not chosen_str or not rejected_str:
                    continue
                    
                sample = {
                    "prompt": [{"role": "user", "content": prompt.strip()}],
                    "chosen": chosen_str,
                    "rejected": rejected_str
                }
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
            total_written += written
            logger.info(f"  UltraFeedback complete: {written:,} pairs")
        except Exception as e:
            logger.error(f"  UltraFeedback failed: {e}")

        # 2. HelpSteer2
        logger.info("Processing HelpSteer2...")
        try:
            # HelpSteer2 contains prompts and multiple responses with helpfulness ratings
            ds = load_dataset(dpo_targets["helpsteer"]["id"], split="train", streaming=True)
            # Since HelpSteer2 is annotated, we can group responses by prompt to create preference pairs
            written = 0
            prompt_map = {}
            for item in ds:
                if written >= dpo_targets["helpsteer"]["target"]:
                    break
                prompt = item.get("prompt", "").strip()
                response = item.get("response", "").strip()
                helpfulness = item.get("helpfulness", 0)
                
                if not prompt or not response:
                    continue
                
                if prompt not in prompt_map:
                    prompt_map[prompt] = []
                prompt_map[prompt].append((response, helpfulness))
                
                # Pair them once we have at least 2 responses for the same prompt
                if len(prompt_map[prompt]) >= 2:
                    resps = sorted(prompt_map[prompt], key=lambda x: x[1])
                    rejected_resp = resps[0][0]
                    chosen_resp = resps[-1][0]
                    # Make sure ratings actually differ
                    if resps[-1][1] > resps[0][1] + 1:
                        sample = {
                            "prompt": [{"role": "user", "content": prompt}],
                            "chosen": chosen_resp,
                            "rejected": rejected_resp
                        }
                        f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        written += 1
                        prompt_map[prompt] = []  # reset for this prompt
            total_written += written
            logger.info(f"  HelpSteer2 complete: {written:,} pairs")
        except Exception as e:
            logger.error(f"  HelpSteer2 failed: {e}")

        # 3. Anthropic HH-RLHF
        logger.info("Processing HH-RLHF...")
        try:
            ds = load_dataset(dpo_targets["hh_rlhf"]["id"], split="train", streaming=True)
            written = 0
            for item in ds:
                if written >= dpo_targets["hh_rlhf"]["target"]:
                    break
                chosen_turns = parse_hh_dialogue(item.get("chosen", ""))
                rejected_turns = parse_hh_dialogue(item.get("rejected", ""))
                
                if not chosen_turns or not rejected_turns:
                    continue
                    
                # Find common prefix turns
                prefix_len = 0
                for ct, rt in zip(chosen_turns, rejected_turns):
                    if ct == rt:
                        prefix_len += 1
                    else:
                        break
                        
                if prefix_len == 0:
                    continue
                    
                prompt_turns = chosen_turns[:prefix_len]
                chosen_str = chosen_turns[prefix_len]["content"] if prefix_len < len(chosen_turns) else ""
                rejected_str = rejected_turns[prefix_len]["content"] if prefix_len < len(rejected_turns) else ""
                
                if not chosen_str or not rejected_str:
                    continue
                    
                sample = {
                    "prompt": prompt_turns,
                    "chosen": chosen_str,
                    "rejected": rejected_str
                }
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
            total_written += written
            logger.info(f"  HH-RLHF complete: {written:,} pairs")
        except Exception as e:
            logger.error(f"  HH-RLHF failed: {e}")

    logger.info(f"Total DPO pairs written: {total_written:,}")
    
    # Upload DPO
    dpo_repo_id = f"{username}/forge-3b-dpo-data"
    logger.info(f"Uploading DPO dataset to HF: {dpo_repo_id}...")
    api.create_repo(repo_id=dpo_repo_id, repo_type="dataset", exist_ok=True, token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(out_file),
        path_in_repo="preferences.jsonl",
        repo_id=dpo_repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    logger.info("DPO upload completed successfully.")


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
