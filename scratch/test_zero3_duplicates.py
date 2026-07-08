import torch
import deepspeed
from config import ForgeModelConfig
from model.forge_model import build_forge_3b

ds_config = {
    "zero_optimization": {
        "stage": 3,
    },
    "train_micro_batch_size_per_gpu": 1,
}

with deepspeed.zero.Init(config_dict_or_path=ds_config):
    model = build_forge_3b(ForgeModelConfig())

print("Checking model.parameters() duplicates:")
params = list(model.parameters())
print(f"Total parameter entries: {len(params)}")
print(f"Unique parameter entries: {len(set(params))}")

ds_ids = {}
duplicates_found = False
for name, p in model.named_parameters():
    ds_id = getattr(p, "ds_id", None)
    if ds_id is not None:
        if ds_id in ds_ids:
            print(f"DUPLICATE DS_ID: {ds_id} shared by '{name}' and '{ds_ids[ds_id]}'")
            duplicates_found = True
        else:
            ds_ids[ds_id] = name

if not duplicates_found:
    print("No duplicate ds_ids found.")
print("Check complete.")
