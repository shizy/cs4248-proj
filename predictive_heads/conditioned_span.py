import torch
import torch.nn as nn
import torch.nn.functional as F
from .head import PredictiveHead, register_head


@register_head("conditioned_span_head")
class ConditionedSpanHead(PredictiveHead):
    """
    Conditioned Span Head modeling P(end | start).
    
    Papers:
      - Yang et al. (2019): "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
        arXiv: https://arxiv.org/abs/1906.08237
      - Joshi et al. (2020): "SpanBERT: Improving Pre-training by Representing and Predicting Spans"
        arXiv: https://arxiv.org/abs/1907.10529
        
    Mechanism:
      Unlike the independent baseline, this head computes start logits first, computes
      a soft-attended start token representation, and conditions end-boundary prediction on
      the start representation using a fused MLP block: [h_i; s; h_i * s].
    """
    def __init__(self, config, dropout_rate=0.1):
        super().__init__(config)
        hidden_size = config.hidden_size

        self.start_linear = nn.Linear(hidden_size, 1)
        self.fuse_linear = nn.Linear(hidden_size * 3, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12))
        self.dropout = nn.Dropout(dropout_rate)
        self.end_linear = nn.Linear(hidden_size, 1)

    def forward(self, seq_output):
        # 1. Compute start logits: (batch_size, seq_len)
        start_logits = self.start_linear(seq_output).squeeze(-1)

        # 2. Extract soft start representation via attention-weighted sum
        start_weights = F.softmax(start_logits, dim=-1).unsqueeze(1)  # (batch_size, 1, seq_len)
        start_repr = torch.bmm(start_weights, seq_output)             # (batch_size, 1, hidden_size)
        start_repr_expanded = start_repr.expand(-1, seq_output.size(1), -1)  # (batch_size, seq_len, hidden_size)

        # 3. Fuse token representation with conditioned start representation: [h; s; h * s]
        fused = torch.cat(
            [seq_output, start_repr_expanded, seq_output * start_repr_expanded],
            dim=-1,
        )
        fused = self.fuse_linear(fused)
        fused = self.activation(fused)
        fused = self.dropout(fused)
        fused = self.layer_norm(fused)

        # 4. Predict end logits conditioned on start: (batch_size, seq_len)
        end_logits = self.end_linear(fused).squeeze(-1)

        return start_logits, end_logits

