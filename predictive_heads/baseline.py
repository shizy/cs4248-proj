import torch.nn as nn
from .head import PredictiveHead, register_head

@register_head("baseline_linear_head")
class BaselineLinearHead(PredictiveHead):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, 2)

    def forward(self, seq_output):
        logits = self.linear(seq_output) 
        start_logits, end_logits = logits.split(1, dim=-1) 
        start_logits = start_logits.squeeze(-1) # shape: (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1) # shape: (batch_size, seq_len)

        return start_logits, end_logits

