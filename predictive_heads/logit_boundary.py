import torch.nn as nn
from .head import PredictiveHead, register_head


@register_head("logit_boundary_head")
class LogitBoundaryHead(PredictiveHead):
    """
    Logit-Space Boundary Convolutional Head (Differentiable Local CRF).

    Papers:
      - Zheng et al. (2015): "Conditional Random Fields as Recurrent Neural Networks"
        arXiv: https://arxiv.org/abs/1502.03240
      - Lample et al. (2016): "Neural Architectures for Named Entity Recognition"
        arXiv: https://arxiv.org/abs/1603.01360

    Mechanism:
      Computes token-level base span logits using a standard linear projection (hidden_size -> 2),
      then applies a zero-initialized 1D convolution directly across the 2-channel logit sequence
      (kernel_size=3, only 12 parameters). This acts as a localized boundary transition filter
      (penalizing or reinforcing boundary shifts) without modifying the 768-dimensional token embeddings.
    """
    def __init__(self, config, kernel_size=3):
        super().__init__(config)
        self.linear = nn.Linear(config.hidden_size, 2)

        # 2 channels (start_logit, end_logit) -> 2 * 2 * 3 = 12 parameters
        self.smoother = nn.Conv1d(
            in_channels=2,
            out_channels=2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        # Zero-init so at step 0 it behaves exactly like the proven baseline_linear_head
        nn.init.zeros_(self.smoother.weight)
        nn.init.zeros_(self.smoother.bias)

    def forward(self, seq_output):
        # 1. Base token-level logits: (batch_size, seq_len, 2)
        base_logits = self.linear(seq_output)

        # 2. Local boundary filter in logit space: (batch_size, 2, seq_len)
        boundary_bias = self.smoother(base_logits.transpose(1, 2)).transpose(1, 2)

        # 3. Combine: base logits + local transition corrections
        final_logits = base_logits + boundary_bias

        start_logits, end_logits = final_logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

