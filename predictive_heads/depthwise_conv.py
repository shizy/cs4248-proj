import torch
import torch.nn as nn
from .head import PredictiveHead, register_head


@register_head("depthwise_conv_head")
class DepthwiseConvHead(PredictiveHead):
    """
    Depthwise Convolutional Boundary Head with ReZero Gating.

    Papers:
      - Bachlechner et al. (2021): "ReZero is All You Need: Fast Convergence at Large Depth"
        arXiv: https://arxiv.org/abs/2003.04887
      - Chollet (2017): "Xception: Deep Learning with Depthwise Separable Convolutions"
        arXiv: https://arxiv.org/abs/1610.02357
      - Yu et al. (2018): "QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension"
        arXiv: https://arxiv.org/abs/1804.09541

    Mechanism:
      Applies depthwise 1D convolution (groups=hidden_size) across the sequence dimension,
      filtering each feature channel independently (only 2,304 parameters vs 1.77M in standard Conv1D).
      Uses a learnable ReZero residual scalar (alpha initialized to 0.0) so that at step 0,
      the head functions as a clean identity passthrough, preventing pretrained representation corruption.
    """
    def __init__(self, config, kernel_size=3, dropout_rate=0.1):
        super().__init__(config)
        hidden_size = config.hidden_size

        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_size,  # Depthwise: each channel is convolved independently
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=getattr(config, "layer_norm_eps", 1e-12))
        self.alpha = nn.Parameter(torch.zeros(1))  # ReZero gate initialized to 0.0
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, seq_output):
        # seq_output: (batch_size, seq_len, hidden_size)
        residual = seq_output
        x = seq_output.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_size)
        x = self.activation(x)
        x = self.dropout(x)

        # ReZero gated residual connection: at step 0, residual + 0 * x = residual
        gated = residual + self.alpha * x
        normed = self.layer_norm(gated)

        logits = self.linear(normed)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

