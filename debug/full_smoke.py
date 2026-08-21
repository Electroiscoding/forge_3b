"""
Full-model smoke test: traces min/max/std/nan through embed -> layer0 -> full forward.
Run: python debug/full_smoke.py
"""
import torch
import os
os.environ["FORGE_NO_TRITON"] = "1"

from config import ForgeModelConfig
from model.forge_model import build_forge_500m


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    cfg = ForgeModelConfig()
    print(f"Building FORGE-500M on {device} ({dtype})...")
    model = build_forge_500m(cfg).to(device).to(dtype)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} ({total_params/1e9:.4f}B)")
    assert total_params <= 1_000_000_000, f"Total parameters {total_params:,} exceeds 1.0B max limit!"

    input_ids = torch.randint(0, cfg.vocab_size, (1, 32), device=device)

    with torch.no_grad():
        h = model.embed_tokens(input_ids)
        print(f"embed: min={h.min().item():.4f} max={h.max().item():.4f} std={h.std().item():.4f}")

        layer0 = model.layers[0]
        out, aux = layer0(
            h, attention_mask=None, position_ids=None,
            position_offset=0, use_cache=False, return_aux_loss=True,
        )
        print(f"after layer 0: min={out.min().item():.4f} max={out.max().item():.4f} "
              f"std={out.std().item():.4f} has_nan={torch.isnan(out).any().item()}")

        result = model(input_ids=input_ids, labels=input_ids, return_aux_loss=True)
        logits = result["logits"]
        print(f"final logits: min={logits.min().item():.4f} max={logits.max().item():.4f} "
              f"std={logits.std().item():.4f}")
        print(f"loss: {result['loss'].item():.4f}")

        w = model.lm_head.weight
        print(f"lm_head weight: min={w.min().item():.4f} max={w.max().item():.4f} std={w.std().item():.4f}")
        print(f"tied to embedding: {model.lm_head.weight is model.embed_tokens.weight}")
        print("\nSMOKE TEST PASSED!")


if __name__ == "__main__":
    main()