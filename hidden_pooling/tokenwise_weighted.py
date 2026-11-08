from .pooler import HiddenStatePooler, register_pooler
import torch
import torch.nn as nn

@register_pooler("tokenwise_weighted")
class TokenwiseWeightedPooler(HiddenStatePooler):
    def __init__(self, config):
        super().__init__(config)
        self.layer_weights = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        
    def forward(self, full_hidden_states: tuple[torch.Tensor], attention_mask: torch.Tensor | None):
        """
        Computes one weight per layer per token per example. Perform weighted average pooling on every layer for each token in an example
        """ 
        stacked_hidden_states = torch.stack(full_hidden_states) # shape: (num_layers, batch_size, seq_len, hidden_size)
        
        scores = self.layer_weights(stacked_hidden_states).squeeze(-1) # shape: (num_layers, batch_size, seq_len)
        weights = torch.softmax(scores, dim=0)

        pooled = (weights.unsqueeze(-1) * stacked_hidden_states).sum(dim=0) # shape: (batch_size, seq_len, hidden_size)

        return pooled

