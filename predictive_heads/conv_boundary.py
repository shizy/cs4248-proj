import torch.nn as nn
from .head import PredictiveHead, register_head


@register_head("conv_boundary_head")
class ConvBoundaryHead(PredictiveHead):
    """
    1D Convolutional Boundary Head for Local N-gram and Phrase Boundary Modeling.
    
    Papers:
      - Yu et al. (2018): "QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension"
        arXiv: https://arxiv.org/abs/1804.09541
      - Choi et al. (2017): "Fast Reading Comprehension with ConvNets"
        arXiv: https://arxiv.org/abs/1711.04352
        
    Mechanism:
      Applies a 1D convolution over sequence token representations with a residual connection
      and LayerNorm. This provides explicit inductive bias for local phrase boundary patterns
      (e.g., adjacent articles, prepositions, and punctuation).
    """
    def __init__(self, config, kernel_size=3, dropout_rate=0.1):
        super().__init__(config)
        hidden_size = config.hidden_size

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12))
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, seq_output):
        # seq_output: (batch_size, seq_len, hidden_size)
        residual = seq_output
        x = seq_output.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_norm(residual + x)

        logits = self.linear(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

