from .pooler import HiddenStatePooler, register_pooler
import torch

@register_pooler("mean")
class MeanPooler(HiddenStatePooler):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, full_hidden_states, attention_mask):
        stacked_hidden_states = torch.stack(full_hidden_states)
        return torch.mean(stacked_hidden_states, dim=0)
