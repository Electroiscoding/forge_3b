"""
FORGE Modeling Class for Hugging Face Transformers
"""

from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from configuration_forge import ForgeConfig
from model.forge_model import ForgeModel


class ForgePreTrainedModel(PreTrainedModel):
    config_class = ForgeConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _no_split_modules = ["ForgeBlock"]
    _tied_weights_keys = ["lm_head.weight"]

    def _init_weights(self, module):
        std = getattr(self.config, "init_std", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class ForgeForCausalLM(ForgePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: ForgeConfig):
        super().__init__(config)
        self.forge_model = ForgeModel(config)
        self.lm_head = self.forge_model.lm_head
        self.embed_tokens = self.forge_model.embed_tokens
        self.layers = self.forge_model.layers
        self.norm_final = self.forge_model.norm_final

    def get_input_embeddings(self):
        return self.forge_model.embed_tokens

    def set_input_embeddings(self, value):
        self.forge_model.embed_tokens = value
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.forge_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.forge_model.lm_head = new_embeddings
        self.lm_head = new_embeddings

    def tie_weights(self):
        if getattr(self.config, "tie_word_embeddings", True):
            self.forge_model.lm_head.weight = self.forge_model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", True)
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)

        res = self.forge_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            use_cache=use_cache,
            return_aux_loss=False,
        )
        loss = res.get("loss")
        logits = res.get("logits")

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=res.get("hidden_states"),
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
