#!/usr/bin/env python3
import os
import argparse
import logging
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Run inference on FORGE-3B using a trained model checkpoint")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory containing config, tokenizer, and weights")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is",
                        help="Prompt to generate text from")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for text generation")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty parameter")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model_dir = Path(args.model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_dir}")
        
    # 1. Load Tokenizer
    from tokenizer.crayon_wrapper import ForgeTokenizer
    tok_dir = model_dir / "tokenizer"
    if tok_dir.exists():
        logger.info(f"Loading tokenizer from {tok_dir}...")
        tokenizer = ForgeTokenizer.from_pretrained(str(tok_dir))
    else:
        logger.info(f"Loading tokenizer from model directory root {model_dir}...")
        tokenizer = ForgeTokenizer.from_pretrained(str(model_dir))
        
    # 2. Load Model Config
    from config import ForgeModelConfig
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find model_config.json or config.json in {model_dir}")
        
    logger.info(f"Loading config from {config_path}")
    model_config = ForgeModelConfig.from_json(str(config_path))
    model_config.vocab_size = tokenizer.vocab_size
    
    # 3. Build Model
    from model.forge_model import build_forge_3b
    logger.info("Building FORGE model...")
    model = build_forge_3b(model_config)
    
    # 4. Load Model Weights
    loaded_weights = False
    for weight_filename in ("model_bf16.pt", "model.pt", "pytorch_model.bin"):
        weight_path = model_dir / weight_filename
        if weight_path.exists():
            logger.info(f"Loading model weights from {weight_path}...")
            state_dict = torch.load(str(weight_path), map_location="cpu")
            # Convert values to bfloat16
            state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            torch.cuda.empty_cache()
            loaded_weights = True
            break
            
    if not loaded_weights:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. "
            "Expected one of: model_bf16.pt, model.pt, pytorch_model.bin"
        )
        
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    
    # 5. Encode prompt
    logger.info(f"Prompt: {args.prompt!r}")
    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True, add_eos=False)], dtype=torch.long, device=device)
    
    # 6. Generate
    logger.info("Generating tokens...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.special_tokens["<|eos|>"],
            pad_token_id=tokenizer.special_tokens["<|pad|>"]
        )
    
    # 7. Decode output
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
    logger.info(f"Generated text:\n{generated_text}")

if __name__ == "__main__":
    main()
