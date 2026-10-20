import torch
import torch.nn as nn
from .head import PredictiveHead, register_head


@register_head("multi_sample_dropout_head")
class MultiSampleDropoutHead(PredictiveHead):
    """
    Multi-Sample Dropout Head for Accelerated Training and Better Generalization.
    
    Paper:
      - Inoue (2019): "Multi-Sample Dropout for Accelerated Training and Better Generalization"
      - arXiv: https://arxiv.org/abs/1905.09788
    
    Mechanism:
      Creates multiple parallel dropout masks during training and passes each through
      a shared projection layer, averaging the resulting logits. During evaluation,
      dropout is an identity operation, introducing zero inference overhead.
    """
    def __init__(self, config, num_samples=5, dropout_rate=0.2):
        super().__init__(config)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_samples)])
        self.linear = nn.Linear(config.hidden_size, 2)

    def forward(self, seq_output):
        if self.training:
            # Pass seq_output through K different dropout samples and average predictions
            logits = torch.mean(
                torch.stack([self.linear(drop(seq_output)) for drop in self.dropouts], dim=0),
                dim=0,
            )
        else:
            logits = self.linear(seq_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

