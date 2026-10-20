# Predictive Heads Reference Guide

This document catalogs the extractive question-answering prediction heads implemented in [`predictive_heads/`](./predictive_heads/).

All heads inherit from [`PredictiveHead`](./predictive_heads/head.py) and are registered via the `@register_head("<name>")` decorator. The registry automatically discovers all head modules via [`predictive_heads/__init__.py`](./predictive_heads/__init__.py), exposing them to `--predictive_head` in [`train.py`](./train.py) and [`train_eval.sh`](./train_eval.sh).

---

## Architecture Summary & Benchmark

| Head Identifier | File | Key Mechanism | arXiv Paper Link | Primary Benefit |
| :--- | :--- | :--- | :--- | :--- |
| `baseline_linear_head` | [`baseline.py`](./predictive_heads/baseline.py) | Single linear projection $\mathbb{R}^H \to \mathbb{R}^2$ | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) | Minimal parameters; standard BERT baseline |
| `multi_sample_dropout_head` | [`multi_sample_dropout.py`](./predictive_heads/multi_sample_dropout.py) | $K$ parallel dropout samples averaged | [arXiv:1905.09788](https://arxiv.org/abs/1905.09788) | Lowers variance, reduces overfitting; 0 eval overhead |
| `residual_mlp_head` | [`residual_mlp.py`](./predictive_heads/residual_mlp.py) | Non-linear GELU MLP with residual connection & LayerNorm | [arXiv:1505.00387](https://arxiv.org/abs/1505.00387) | Higher expressive capacity without vanishing gradients |
| `conv_boundary_head` | [`conv_boundary.py`](./predictive_heads/conv_boundary.py) | 1D Local Convolution + LayerNorm residual | [arXiv:1804.09541](https://arxiv.org/abs/1804.09541) | Enhances local n-gram boundary awareness (boosts EM) |
| `conditioned_span_head` | [`conditioned_span.py`](./predictive_heads/conditioned_span.py) | Soft-attended start fusion for $P(\text{end} \mid \text{start})$ | [arXiv:1906.08237](https://arxiv.org/abs/1906.08237) | Enforces span coherence; eliminates independent head mismatch |

---

## Head Details

### 1. Baseline Linear Head (`baseline_linear_head`)
* **File**: [`predictive_heads/baseline.py`](./predictive_heads/baseline.py)
* **Paper**: *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* (Devlin et al., 2018)
* **arXiv Link**: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
* **Mechanism**:
  Takes the final token hidden states $H \in \mathbb{R}^{B \times L \times D}$ from BERT and applies a single linear projection:
  $$\text{logits} = H W + b, \quad W \in \mathbb{R}^{D \times 2}$$
  Splits into $\text{start\_logits}, \text{end\_logits} \in \mathbb{R}^{B \times L}$.
* **Usage**:
  ```bash
  python train.py --predictive_head baseline_linear_head
  ```

---

### 2. Multi-Sample Dropout Head (`multi_sample_dropout_head`)
* **File**: [`predictive_heads/multi_sample_dropout.py`](./predictive_heads/multi_sample_dropout.py)
* **Paper**: *Multi-Sample Dropout for Accelerated Training and Better Generalization* (Inoue, 2019)
* **arXiv Link**: [https://arxiv.org/abs/1905.09788](https://arxiv.org/abs/1905.09788)
* **Mechanism**:
  Creates $K$ independent dropout masks (default $K=5$, $p=0.2$) applied to the same token representation $H$. Each dropped representation is projected through the shared linear weight matrix $W$, and predictions are averaged:
  $$\text{logits}_{\text{train}} = \frac{1}{K} \sum_{k=1}^K \text{Dropout}_k(H) W + b$$
  During evaluation (`model.eval()`), dropout behaves as an identity map, collapsing back to standard single linear evaluation with zero computational or latency overhead.
* **Why it helps**:
  Acts as an implicit ensemble within a single model. It accelerates training convergence, reduces gradient variance, and regularly yields $+0.3$ to $+0.6$ F1 gains on reading comprehension benchmarks.
* **Usage**:
  ```bash
  python train.py --predictive_head multi_sample_dropout_head
  ```

---

### 3. Residual MLP Head (`residual_mlp_head`)
* **File**: [`predictive_heads/residual_mlp.py`](./predictive_heads/residual_mlp.py)
* **Papers**:
  - *Highway Networks* (Srivastava et al., 2015) — [https://arxiv.org/abs/1505.00387](https://arxiv.org/abs/1505.00387)
  - *Deep Residual Learning for Image Recognition* (He et al., 2016) — [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
* **Mechanism**:
  Rather than directly mapping token embeddings to logits linearly, it passes token states through a non-linear feedforward layer with residual connection and Layer Normalization:
  $$\tilde{H} = \text{LayerNorm}(H + \text{Dropout}(\text{GELU}(H W_1 + b_1)))$$
  $$\text{logits} = \tilde{H} W_2 + b_2$$
* **Why it helps**:
  A standard single linear layer has limited representational capacity to separate boundary cases. Pure MLPs without residual connections can destabilize fine-tuning; the residual bypass ensures gradients flow directly while providing additional non-linear capacity.
* **Usage**:
  ```bash
  python train.py --predictive_head residual_mlp_head
  ```

---

### 4. Convolutional Boundary Head (`conv_boundary_head`)
* **File**: [`predictive_heads/conv_boundary.py`](./predictive_heads/conv_boundary.py)
* **Papers**:
  - *QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension* (Yu et al., 2018) — [https://arxiv.org/abs/1804.09541](https://arxiv.org/abs/1804.09541)
  - *Fast Reading Comprehension with ConvNets* (Choi et al., 2017) — [https://arxiv.org/abs/1711.04352](https://arxiv.org/abs/1711.04352)
* **Mechanism**:
  Applies a 1D convolution ($k=3$, padding $=1$) across the token sequence dimension, followed by GELU, Dropout, LayerNorm, and residual connection:
  $$H_{\text{conv}} = \text{Conv1D}(H^T)^T$$
  $$\tilde{H} = \text{LayerNorm}(H + \text{Dropout}(\text{GELU}(H_{\text{conv}})))$$
  $$\text{logits} = \tilde{H} W + b$$
* **Why it helps**:
  While BERT self-attention captures long-range dependencies, span boundaries are heavily governed by local syntax (e.g. adjacent prepositions, articles, punctuation, and capitalization). The 1D convolution injects a localized n-gram inductive bias, improving Exact Match (EM) boundary precision.
* **Usage**:
  ```bash
  python train.py --predictive_head conv_boundary_head
  ```

---

### 5. Conditioned Span Head (`conditioned_span_head`)
* **File**: [`predictive_heads/conditioned_span.py`](./predictive_heads/conditioned_span.py)
* **Papers**:
  - *XLNet: Generalized Autoregressive Pretraining for Language Understanding* (Yang et al., 2019) — [https://arxiv.org/abs/1906.08237](https://arxiv.org/abs/1906.08237)
  - *SpanBERT: Improving Pre-training by Representing and Predicting Spans* (Joshi et al., 2020) — [https://arxiv.org/abs/1907.10529](https://arxiv.org/abs/1907.10529)
* **Mechanism**:
  Standard extractive QA models make the independence assumption $P(\text{start}, \text{end}) = P(\text{start}) \cdot P(\text{end})$. This head models the joint probability autoregressively as $P(\text{start}) \cdot P(\text{end} \mid \text{start})$:
  1. Compute start logits: $\text{start\_logits} = H W_{\text{start}}$
  2. Compute attention-weighted soft start representation:
     $$\alpha = \text{softmax}(\text{start\_logits}), \quad s = \sum_{i=1}^L \alpha_i H_i \in \mathbb{R}^{B \times D}$$
  3. Fuse each token state $H_i$ with the start summary $s$ using concatenation and element-wise interaction:
     $$F_i = [H_i \,;\, s \,;\, H_i \odot s]$$
     $$\tilde{H}_i = \text{LayerNorm}(\text{Dropout}(\text{GELU}(F_i W_{\text{fuse}})))$$
  4. Predict end logits conditioned on start: $\text{end\_logits}_i = \tilde{H}_i W_{\text{end}}$
* **Why it helps**:
  Prevents predicted end positions that are semantically incongruent or incompatible with the chosen start position.
* **Usage**:
  ```bash
  python train.py --predictive_head conditioned_span_head
  ```

---

## Running with SLURM Cluster Batch Script

To evaluate any head end-to-end on the compute cluster via [`train_eval.sh`](./train_eval.sh):

```bash
# Example: Multi-Sample Dropout
sbatch train_eval.sh --predictive_head multi_sample_dropout_head

# Example: Conditioned Span Head
sbatch train_eval.sh --predictive_head conditioned_span_head

# Example: Convolutional Boundary Head
sbatch train_eval.sh --predictive_head conv_boundary_head

# Example: Residual MLP Head
sbatch train_eval.sh --predictive_head residual_mlp_head
```

The model checkpoint, tokenizer, hyperparameters (`training_args.json`), predicted spans (`predictions.json`), and evaluation metrics (`eval_results.json`) will all be automatically stored in `checkpoints/<job_id>/`.
