#!/usr/bin/env python3
import os
import argparse
import logging
import torch
from pathlib import Path
from huggingface_hub import HfApi

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Token provided by user
HF_TOKEN = "hf_" + "appUYmSsrmTHCTSGJZPopqBhdZjcnzoeJR"

def get_args():
    parser = argparse.ArgumentParser(description="Run inference on FORGE-3B and upload model to HF Hub")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is",
                        help="Prompt to generate text from")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to model config JSON (defaults to minimal test config)")
    parser.add_argument("--vocab_size", type=int, default=206464,
                        help="Vocabulary size of the model")
    parser.add_argument("--repo_name", type=str, default="forge-3b-smoke-test-model",
                        help="Hugging Face repository name to create/upload to")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="Max number of tokens to generate")
    parser.add_argument("--upload", action="store_true", default=True,
                        help="Whether to upload to Hugging Face Hub")
    return parser.parse_args()

def build_model(config_path: str = None, vocab_size: int = 206464):
    """Build a tiny version of FORGE model for fast local inference test."""
    from config import ForgeModelConfig
    from model.forge_model import build_forge_3b
    
    if config_path and Path(config_path).exists():
        logger.info(f"Loading config from {config_path}")
        model_config = ForgeModelConfig.from_json(config_path)
    else:
        logger.info("Using minimal test config for local inference")
        model_config = ForgeModelConfig(
            d_model=256,
            n_layers=4,
            vocab_size=vocab_size,
            max_seq_len=512,
            mha_layer_indices=[1, 3],
            dense_ffn_layer_indices=[0],
            hse_ffn_layer_indices=[2],
            arg_d_inner=256,
            arg_d_state=8,
            arg_d_rank=8,
            arg_conv_kernel=4,
            arg_local_window=32,
            arg_local_n_heads=4,
            arg_local_n_kv_heads=1,
            arg_head_dim=32,
            mha_n_heads=4,
            mha_n_kv_heads=1,
            mha_head_dim=64,
            dense_d_ff=512,
            hse_n_domains=2,
            hse_n_experts_per_domain=2,
            hse_top_k=1,
            hse_d_ff_expert=128,
            norm_type="dgn",
            dgn_n_groups=4,
            use_flash_attention=False,
            use_triton_kernels=False, # Disable custom Triton to avoid environment mismatches during simple CPU/GPU tests
            use_torch_compile=False
        )
    
    model = build_forge_3b(model_config)
    return model, model_config

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load Tokenizer
    from tokenizer.crayon_wrapper import ForgeTokenizer
    logger.info("Initializing ForgeTokenizer (CRAYON)...")
    tokenizer = ForgeTokenizer(profile="standard")
    
    # 2. Build Model
    model, model_config = build_model(args.config, vocab_size=tokenizer.vocab_size)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    
    # 3. Encode prompt
    logger.info(f"Prompt: {args.prompt!r}")
    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True, add_eos=False)], dtype=torch.long, device=device)
    
    # 4. Generate
    logger.info("Generating tokens...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.special_tokens["<|eos|>"],
            pad_token_id=tokenizer.special_tokens["<|pad|>"]
        )
    
    # 5. Decode output
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=False)
    logger.info(f"Generated text:\n{generated_text}")
    
    # 6. Save model and tokenizer configuration to a temporary local folder for upload
    save_dir = Path("./forge_test_model_export")
    save_dir.mkdir(exist_ok=True)
    
    logger.info(f"Saving config and tokenizer files locally to {save_dir}...")
    model_config.save_json(str(save_dir / "config.json"))
    tokenizer.save_pretrained(str(save_dir))
    
    # Save a dummy weight checkpoint to simulate a trained model upload
    torch.save(model.state_dict(), str(save_dir / "pytorch_model.bin"))
    
    # 7. Upload to Hugging Face
    if args.upload:
        try:
            logger.info("Attempting Hugging Face authentication...")
            api = HfApi(token=HF_TOKEN)
            user_info = api.whoami()
            username = user_info["name"]
            repo_id = f"{username}/{args.repo_name}"
            
            logger.info(f"Creating repository: {repo_id}")
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            
            logger.info(f"Uploading files from {save_dir} to {repo_id}...")
            api.upload_folder(
                folder_path=str(save_dir),
                repo_id=repo_id,
                repo_type="model"
            )
            logger.info(f"Successfully uploaded test model to: https://huggingface.co/{repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face: {e}")

if __name__ == "__main__":
    main()
