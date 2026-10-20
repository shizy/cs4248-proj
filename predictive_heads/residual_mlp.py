import torch.nn as nn
from .head import PredictiveHead, register_head


@register_head("residual_mlp_head")
class ResidualMLPHead(PredictiveHead):
    """
    Residual Non-Linear MLP Head with LayerNorm and GELU Activation.
    
    Papers:
      - Srivastava et al. (2015): "Highway Networks"
        arXiv: https://arxiv.org/abs/1505.00387
      - Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
        arXiv: https://arxiv.org/abs/1810.04805
        
    Mechanism:
      Adds a residual feed-forward block: LayerNorm(seq_output + Dropout(GELU(Linear(seq_output))))
      before projecting into start and end span logits. The skip connection preserves
      pretrained representations while granting non-linear decision capacity.
    """
    def __init__(self, config, intermediate_size=None, dropout_rate=0.1):
        super().__init__(config)
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or hidden_size

        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12))
        self.out_proj = nn.Linear(intermediate_size, 2)

    def forward(self, seq_output):
        residual = seq_output
        hidden = self.dense(seq_output)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        normed = self.layer_norm(residual + hidden)

        logits = self.out_proj(normed)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

