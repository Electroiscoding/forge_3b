import torch
import torch.nn as nn
import deepspeed

ds_config = {
    "zero_optimization": {
        "stage": 3,
    },
    "train_micro_batch_size_per_gpu": 1,
}

class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 10)
        self.lm_head = nn.Linear(10, 10, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

with deepspeed.zero.Init(config_dict_or_path=ds_config):
    model = TiedModel()

print("Parameters in model:")
for name, p in model.named_parameters():
    print(f"Name: {name}, ds_id: {getattr(p, 'ds_id', None)}, shape: {p.shape}")
