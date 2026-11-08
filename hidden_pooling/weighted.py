from .pooler import HiddenStatePooler, register_pooler
import torch
import torch.nn as nn

@register_pooler("weighted")
class WeightedPooler(HiddenStatePooler):
    def __init__(self, config):
        super().__init__(config)
        self.layer_weights = nn.Sequential(
            nn.Linear(config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        
    def forward(self, full_hidden_states: tuple[torch.Tensor], attention_mask: torch.Tensor | None):
        """
        Computes one weight per layer per example. Perform weighted average pooling on every layer for each example
        """ 
        stacked_hidden_states = torch.stack(full_hidden_states) # shape: (num_layers, batch_size, seq_len, hidden_size)
        
        if attention_mask is None:
            summaries = stacked_hidden_states.mean(dim=2) # mean across sequence length
        else:
            mask = attention_mask.to(
                device=stacked_hidden_states.device,
                dtype=stacked_hidden_states.dtype
            )[None, :, :, None] # shape: (1, batch_size, seq_len, 1)

            summaries = (stacked_hidden_states * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1) # shape: (num_layers, batch_size, hidden_size

        scores = self.layer_weights(summaries).squeeze(-1) # shape: (num_layers, batch_size)
        weights = torch.softmax(scores, dim=0)

        pooled = (stacked_hidden_states * weights[:, :, None, None]).sum(dim=0) # shape: (batch_size, seq_len, hidden_size)

        return pooled

