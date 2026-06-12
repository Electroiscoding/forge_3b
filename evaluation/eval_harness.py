"""
FORGE-3B Evaluation Harness.

Runs a battery of standard LM benchmarks against any FORGE-3B checkpoint
and reports results in a structured JSON + Markdown table.

Supported benchmark categories:
  1. Language Modelling — perplexity on held-out validation splits
  2. MMLU              — zero-shot and 5-shot accuracy
  3. HellaSwag         — zero-shot multiple-choice
  4. HumanEval         — code generation pass@1 / pass@10
  5. GSM8K             — grade-school math, chain-of-thought 0-shot
  6. TruthfulQA        — truthfulness / hallucination rate

Usage:
    python evaluation/eval_harness.py \\
        --model_path /workspace/checkpoints/forge_3b_sft/final \\
        --tasks      all \\
        --output     /workspace/eval_results/sft_eval.json \\
        --device     cuda

For a quick smoke test (fast subset):
    python evaluation/eval_harness.py \\
        --model_path ./checkpoints/forge_3b_pretrain/final \\
        --tasks      ppl,hellaswag \\
        --n_few_shot 0 \\
        --max_samples 500 \\
        --device cuda
"""

from __future__ import annotations

import os
import gc
import re
import sys
import json
import time
import math
import logging
import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger("eval_harness")

# ─────────────────────────────────────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_forge_model(
    model_path: str,
    device: torch.device,
    bf16: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a FORGE-3B model + tokenizer from a checkpoint directory.
    Returns (model, tokenizer).
    """
    model_dir = Path(model_path)

    # Model config
    cfg_path = model_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"model_config.json not found in {model_dir}")

    from config import ForgeModelConfig
    model_config = ForgeModelConfig.from_json(str(cfg_path))

    # Tokenizer
    tok_dir = model_dir / "tokenizer"
    from tokenizer.crayon_wrapper import ForgeTokenizer
    if tok_dir.exists():
        tokenizer = ForgeTokenizer.from_pretrained(str(tok_dir))
    else:
        tokenizer = ForgeTokenizer(profile="standard", device="cpu")
        logger.warning(f"No tokenizer dir found, using default profile")
    model_config.vocab_size = tokenizer.vocab_size

    # Model
    from model.forge_model import build_forge_3b
    model = build_forge_3b(model_config)

    # Weights
    dtype = torch.bfloat16 if bf16 else torch.float32
    for weight_name in ("model_bf16.pt", "model.pt", "pytorch_model.bin"):
        w_path = model_dir / weight_name
        if w_path.exists():
            state_dict = torch.load(str(w_path), map_location="cpu")
            state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            torch.cuda.empty_cache()
            logger.info(f"Weights loaded from {w_path}")
            break
    else:
        raise FileNotFoundError(f"No weights found in {model_dir}")

    model = model.to(device=device, dtype=dtype)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {n_params / 1e9:.3f}B params | dtype={dtype} | device={device}")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity Evaluator
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_perplexity(
    model,
    tokenizer,
    data_dir: str,
    device: torch.device,
    seq_len: int = 2048,
    max_samples: int = 2000,
    domain: str = "validation",
) -> Dict[str, float]:
    """
    Compute per-token perplexity on held-out validation shards.
    Uses memory-mapped numpy arrays for zero-copy loading.
    """
    from data.dataset import PackedTokenDataset

    data_path = Path(data_dir)
    if not data_path.exists():
        return {"ppl": float("nan"), "n_samples": 0, "error": "data_dir not found"}

    try:
        dataset = PackedTokenDataset(
            data_dir=str(data_path),
            seq_len=seq_len,
            split="val",
            max_samples=max_samples,
            shuffle_files=False,
        )
    except FileNotFoundError:
        return {"ppl": float("nan"), "n_samples": 0, "error": "no val shards"}

    n_samples   = min(len(dataset), max_samples)
    total_nll   = 0.0
    total_tokens = 0

    for i in range(n_samples):
        sample    = dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)  # (1, T)
        labels    = input_ids.clone()

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)

        loss = outputs["loss"].item()
        # loss = mean NLL over non-ignored tokens; seq_len - 1 labels (shifted)
        total_nll    += loss * (seq_len - 1)
        total_tokens += (seq_len - 1)

        if i % 200 == 0:
            running_ppl = math.exp(total_nll / max(total_tokens, 1))
            logger.info(f"  PPL [{domain}]: {i}/{n_samples} → running ppl={running_ppl:.2f}")

    ppl = math.exp(total_nll / max(total_tokens, 1))
    logger.info(f"  PPL [{domain}]: {ppl:.3f} ({n_samples} samples)")
    return {
        "ppl":      round(ppl, 4),
        "nll":      round(total_nll / max(total_tokens, 1), 6),
        "n_samples": n_samples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multiple-Choice Evaluator (MMLU, HellaSwag, TruthfulQA)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_multiple_choice(
    model,
    tokenizer,
    examples: List[Dict],
    device: torch.device,
    n_few_shot: int = 0,
) -> Dict[str, float]:
    """
    Log-likelihood based multiple-choice evaluation.

    Each example must have:
      'question': str
      'choices':  List[str]    (A, B, C, D)
      'answer':   int          (0-indexed correct choice)
      'context':  str (optional, for few-shot examples)

    Returns accuracy and per-category breakdown if 'category' field present.
    """
    correct = 0
    total   = 0
    category_stats: Dict[str, Dict] = {}

    few_shot_prefix = ""
    if n_few_shot > 0 and len(examples) > n_few_shot:
        fs_examples = examples[:n_few_shot]
        fs_parts = []
        for ex in fs_examples:
            choice_labels = "ABCD"
            q = ex["question"]
            choices = "\n".join(
                f"{choice_labels[j]}. {c}" for j, c in enumerate(ex["choices"])
            )
            answer_label = choice_labels[ex["answer"]]
            fs_parts.append(f"Q: {q}\n{choices}\nAnswer: {answer_label}")
        few_shot_prefix = "\n\n".join(fs_parts) + "\n\n"
        examples = examples[n_few_shot:]

    for ex in examples:
        question = ex["question"]
        choices  = ex["choices"]
        answer   = ex["answer"]
        category = ex.get("category", "general")

        choice_labels = "ABCD"[:len(choices)]
        prompt = few_shot_prefix + f"Q: {question}\n"
        for j, c in enumerate(choices):
            prompt += f"{choice_labels[j]}. {c}\n"
        prompt += "Answer:"

        log_likelihoods = []
        for j, choice in enumerate(choices):
            completion = f" {choice_labels[j]}"
            full_text  = prompt + completion

            try:
                input_ids = torch.tensor(
                    tokenizer.encode(full_text, add_bos=True, add_eos=False),
                    dtype=torch.long,
                ).unsqueeze(0).to(device)

                prompt_ids = torch.tensor(
                    tokenizer.encode(prompt, add_bos=True, add_eos=False),
                    dtype=torch.long,
                ).unsqueeze(0).to(device)

                prompt_len = prompt_ids.shape[1]
            except Exception:
                log_likelihoods.append(-1e9)
                continue

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids)

            logits   = outputs["logits"]                          # (1, T, V)
            # Only score the completion tokens
            comp_logits  = logits[0, prompt_len - 1 : -1, :]    # (n_comp, V)
            comp_ids     = input_ids[0, prompt_len:]             # (n_comp,)

            if comp_ids.numel() == 0:
                log_likelihoods.append(-1e9)
                continue

            lp = F.log_softmax(comp_logits.float(), dim=-1)
            ll = lp[range(len(comp_ids)), comp_ids].sum().item()
            log_likelihoods.append(ll)

        pred    = int(torch.tensor(log_likelihoods).argmax().item())
        is_corr = (pred == answer)

        correct += int(is_corr)
        total   += 1

        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        category_stats[category]["correct"] += int(is_corr)
        category_stats[category]["total"]   += 1

    accuracy = correct / max(total, 1)
    logger.info(f"  MC accuracy: {accuracy:.4f} ({correct}/{total})")

    result = {
        "accuracy":   round(accuracy, 4),
        "n_correct":  correct,
        "n_total":    total,
        "n_few_shot": n_few_shot,
    }
    if len(category_stats) > 1:
        result["per_category"] = {
            cat: round(v["correct"] / max(v["total"], 1), 4)
            for cat, v in category_stats.items()
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Code Evaluation (HumanEval) — requires `human_eval` package
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_humaneval(
    model,
    tokenizer,
    device: torch.device,
    n_samples: int = 20,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    HumanEval pass@1 and pass@10 evaluation.
    Requires: pip install human-eval
    """
    try:
        from human_eval.data import write_jsonl, read_problems
        from human_eval.evaluation import evaluate_functional_correctness
    except ImportError:
        logger.warning("human-eval not installed — skipping HumanEval. "
                       "Install with: pip install human-eval")
        return {"pass@1": float("nan"), "pass@10": float("nan"), "error": "not installed"}

    problems = read_problems()
    samples  = []

    for task_id, problem in problems.items():
        prompt_ids = tokenizer.encode(problem["prompt"], add_bos=True, add_eos=False)
        input_ids  = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        for _ in range(n_samples):
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                generated = _greedy_or_sample(
                    model=model,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    eos_token_id=tokenizer.eos_token_id,
                )

            completion = tokenizer.decode(generated)
            # Truncate at the first top-level def/class that would break the solution
            completion = _truncate_completion(completion)
            samples.append({"task_id": task_id, "completion": completion})

    # Write to temp file and evaluate
    samples_file = "/tmp/forge_humaneval_samples.jsonl"
    write_jsonl(samples_file, samples)

    try:
        results = evaluate_functional_correctness(samples_file)
        logger.info(f"  HumanEval: pass@1={results.get('pass@1', '?'):.4f}")
        return {k: round(float(v), 4) for k, v in results.items()}
    except Exception as exc:
        logger.error(f"HumanEval execution failed: {exc}")
        return {"error": str(exc)}


def _truncate_completion(code: str) -> str:
    """Truncate a generated code completion at a natural boundary."""
    stop_sequences = ["\nclass ", "\ndef ", "\n#", "\nif __name__"]
    for stop in stop_sequences:
        idx = code.find(stop)
        if idx != -1:
            return code[:idx]
    return code


def _greedy_or_sample(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    eos_token_id: int = 2,
) -> List[int]:
    """
    Autoregressive generation: greedy (temp=0) or temperature sampling.
    Returns only the newly generated token ids (not the prompt).
    """
    generated: List[int] = []
    current_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=current_ids)
        next_logits = outputs["logits"][:, -1, :].float()

        if temperature <= 0.0 or temperature == 0:
            next_token = next_logits.argmax(dim=-1)
        else:
            probs      = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        token_id = next_token.item()
        if token_id == eos_token_id:
            break
        generated.append(token_id)
        current_ids = torch.cat(
            [current_ids, next_token.unsqueeze(0)], dim=1
        )

    return generated


# ─────────────────────────────────────────────────────────────────────────────
# GSM8K Evaluation (chain-of-thought math reasoning)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_gsm8k(
    model,
    tokenizer,
    gsm8k_path: str,
    device: torch.device,
    max_samples: int = 500,
    max_new_tokens: int = 256,
) -> Dict[str, float]:
    """
    Evaluate on GSM8K grade-school math.
    Expects a JSONL file with 'question' and 'answer' fields.
    """
    path = Path(gsm8k_path)
    if not path.exists():
        return {"accuracy": float("nan"), "error": "gsm8k file not found"}

    problems = []
    with open(str(path)) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))

    problems = problems[:max_samples]
    correct  = 0

    _ANS_RE = re.compile(r"####\s*(-?[\d,]+)")

    for prob in problems:
        question = prob["question"]
        ref_ans  = prob["answer"]
        ref_num  = _extract_gsm8k_number(ref_ans, _ANS_RE)

        prompt = (
            "Solve the math problem step by step, then state the final answer "
            "after '####'.\n\n"
            f"Problem: {question}\nSolution:"
        )

        prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids  = torch.tensor([prompt_ids], dtype=torch.long).to(device)

        generated = _greedy_or_sample(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(generated)
        pred_num = _extract_gsm8k_number(decoded, _ANS_RE)

        if ref_num is not None and pred_num is not None and abs(ref_num - pred_num) < 1e-6:
            correct += 1

    accuracy = correct / max(len(problems), 1)
    logger.info(f"  GSM8K: {accuracy:.4f} ({correct}/{len(problems)})")
    return {"accuracy": round(accuracy, 4), "n_correct": correct, "n_total": len(problems)}


def _extract_gsm8k_number(text: str, pattern: re.Pattern) -> Optional[float]:
    match = pattern.search(text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────────────────────────────────────

class EvalHarness:
    """Orchestrates all evaluation tasks and writes results."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        bf16: bool = True,
        max_samples: int = 2000,
        n_few_shot: int = 5,
        output_path: Optional[str] = None,
        data_base: str = "./data",
    ):
        self.device      = device
        self.max_samples = max_samples
        self.n_few_shot  = n_few_shot
        self.output_path = output_path
        self.data_base   = Path(data_base)

        logger.info(f"Loading model from {model_path}...")
        self.model, self.tokenizer = load_forge_model(
            model_path, device=device, bf16=bf16
        )
        self.results: Dict[str, Any] = {
            "model_path":   model_path,
            "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
            "device":       str(device),
            "n_few_shot":   n_few_shot,
            "max_samples":  max_samples,
            "tasks":        {},
        }

    def run_ppl(self):
        """Perplexity on held-out validation sets."""
        logger.info("── Perplexity Evaluation ───────────────────────────────")
        ppl_results = {}

        # Try to evaluate on each available domain val split
        for domain in ["wikipedia", "openwebmath", "fineweb_edu", "arxiv", "books"]:
            domain_dir = self.data_base / "tokenized" / domain
            if domain_dir.exists():
                logger.info(f"  PPL [{domain}]...")
                r = evaluate_perplexity(
                    self.model, self.tokenizer,
                    str(domain_dir), self.device,
                    seq_len=2048,
                    max_samples=min(self.max_samples, 500),
                    domain=domain,
                )
                ppl_results[domain] = r

        self.results["tasks"]["perplexity"] = ppl_results

    def run_mmlu(self, mmlu_path: Optional[str] = None):
        """MMLU 5-shot accuracy."""
        logger.info("── MMLU Evaluation ─────────────────────────────────────")

        path = mmlu_path or str(self.data_base / "eval" / "mmlu.jsonl")
        if not Path(path).exists():
            logger.warning(f"MMLU data not found at {path} — skipping")
            self.results["tasks"]["mmlu"] = {"error": "data not found", "path": path}
            return

        examples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        examples = examples[:self.max_samples]
        result   = evaluate_multiple_choice(
            self.model, self.tokenizer, examples,
            device=self.device, n_few_shot=self.n_few_shot,
        )
        self.results["tasks"]["mmlu"] = result
        logger.info(f"  MMLU: accuracy={result['accuracy']:.4f}")

    def run_hellaswag(self, hs_path: Optional[str] = None):
        """HellaSwag 0-shot accuracy."""
        logger.info("── HellaSwag Evaluation ────────────────────────────────")

        path = hs_path or str(self.data_base / "eval" / "hellaswag.jsonl")
        if not Path(path).exists():
            logger.warning(f"HellaSwag data not found at {path} — skipping")
            self.results["tasks"]["hellaswag"] = {"error": "data not found"}
            return

        examples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        examples = examples[:self.max_samples]
        result   = evaluate_multiple_choice(
            self.model, self.tokenizer, examples,
            device=self.device, n_few_shot=0,
        )
        self.results["tasks"]["hellaswag"] = result
        logger.info(f"  HellaSwag: accuracy={result['accuracy']:.4f}")

    def run_gsm8k(self, gsm8k_path: Optional[str] = None):
        """GSM8K chain-of-thought math accuracy."""
        logger.info("── GSM8K Evaluation ────────────────────────────────────")

        path = gsm8k_path or str(self.data_base / "eval" / "gsm8k_test.jsonl")
        result = evaluate_gsm8k(
            self.model, self.tokenizer,
            gsm8k_path=path, device=self.device,
            max_samples=min(self.max_samples, 500),
        )
        self.results["tasks"]["gsm8k"] = result

    def run_humaneval(self):
        """HumanEval pass@1."""
        logger.info("── HumanEval Evaluation ────────────────────────────────")
        result = evaluate_humaneval(
            self.model, self.tokenizer, device=self.device,
            n_samples=10, max_new_tokens=512, temperature=0.2,
        )
        self.results["tasks"]["humaneval"] = result

    def run_all(self, tasks: str = "all"):
        """Run all or a comma-separated subset of tasks."""
        task_set = set(tasks.lower().split(",")) if tasks != "all" else {
            "ppl", "mmlu", "hellaswag", "gsm8k", "humaneval"
        }

        if "ppl" in task_set:
            self.run_ppl()
        if "mmlu" in task_set:
            self.run_mmlu()
        if "hellaswag" in task_set:
            self.run_hellaswag()
        if "gsm8k" in task_set:
            self.run_gsm8k()
        if "humaneval" in task_set:
            self.run_humaneval()

        self._save_results()
        self._print_summary()

    def _save_results(self):
        if self.output_path:
            out = Path(self.output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(str(out), "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved: {out}")

    def _print_summary(self):
        tasks = self.results["tasks"]

        print("\n" + "=" * 64)
        print(" FORGE-3B EVALUATION SUMMARY")
        print("=" * 64)

        # PPL
        if "perplexity" in tasks:
            print("\n📊 Perplexity (↓ better)")
            for domain, r in tasks["perplexity"].items():
                if isinstance(r, dict) and "ppl" in r:
                    print(f"   {domain:<20}: PPL = {r['ppl']:.2f}")

        # Accuracy benchmarks
        acc_tasks = {
            "mmlu":       "MMLU 5-shot",
            "hellaswag":  "HellaSwag 0-shot",
            "gsm8k":      "GSM8K Math",
            "humaneval":  "HumanEval pass@1",
        }
        print("\n🎯 Accuracy Benchmarks (↑ better)")
        for key, name in acc_tasks.items():
            if key in tasks:
                r = tasks[key]
                if "accuracy" in r:
                    print(f"   {name:<25}: {r['accuracy']:.4f}  ({r.get('n_correct','?')}/{r.get('n_total','?')})")
                elif "pass@1" in r:
                    print(f"   {name:<25}: pass@1={r['pass@1']:.4f}")
                elif "error" in r:
                    print(f"   {name:<25}: SKIPPED ({r['error']})")

        print("=" * 64)
        print(f"  Model  : {self.results['model_path']}")
        print(f"  Saved  : {self.output_path or '<not saved>'}")
        print("=" * 64 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="FORGE-3B Evaluation Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to checkpoint dir (must contain model_config.json + weights)")
    parser.add_argument("--tasks",      default="all",
                        help="Comma-separated tasks: ppl,mmlu,hellaswag,gsm8k,humaneval — or 'all'")
    parser.add_argument("--output",     default=None,
                        help="Path to write JSON results (optional)")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Maximum samples per task (for fast iteration)")
    parser.add_argument("--n_few_shot",  type=int, default=5)
    parser.add_argument("--data_base",   default="./data",
                        help="Base dir containing eval/ and tokenized/ subdirectories")
    parser.add_argument("--no_bf16",     action="store_true",
                        help="Disable BF16 (use FP32 — slower but higher precision)")
    args = parser.parse_args()

    device = torch.device(args.device)
    harness = EvalHarness(
        model_path=args.model_path,
        device=device,
        bf16=not args.no_bf16,
        max_samples=args.max_samples,
        n_few_shot=args.n_few_shot,
        output_path=args.output,
        data_base=args.data_base,
    )
    harness.run_all(tasks=args.tasks)
