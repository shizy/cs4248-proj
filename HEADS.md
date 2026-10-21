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
| `conv_boundary_head` | [`conv_boundary.py`](./predictive_heads/conv_boundary.py) | Dense 1D Conv ($768 \to 768$) + LayerNorm residual | [arXiv:1804.09541](https://arxiv.org/abs/1804.09541) | Explores boundary n-grams (note: vulnerable to un-gated 1.77M param collapse) |
| `conditioned_span_head` | [`conditioned_span.py`](./predictive_heads/conditioned_span.py) | Soft-attended start fusion for $P(\text{end} \mid \text{start})$ | [arXiv:1906.08237](https://arxiv.org/abs/1906.08237) | Enforces span coherence; eliminates independent head mismatch |
| `depthwise_conv_head` | [`depthwise_conv.py`](./predictive_heads/depthwise_conv.py) | Depthwise Conv1D (`groups=H`) + ReZero gate ($\alpha=0$) | [arXiv:2003.04887](https://arxiv.org/abs/2003.04887) | Fixes conv collapse: 2.3k params; preserves pretrained representations at step 0 |
| `logit_boundary_head` | [`logit_boundary.py`](./predictive_heads/logit_boundary.py) | 1D Boundary Conv on Logits ($2 \to 2$ channels) | [arXiv:1502.03240](https://arxiv.org/abs/1502.03240) | Differentiable local boundary filter in score space (12 params) |

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
  Applies a dense 1D convolution ($k=3$, padding $=1$) across the token sequence dimension, followed by GELU, Dropout, LayerNorm, and residual connection.
* **Analysis**:
  Contains 1.77M randomly initialized weights without zero-gating, which causes representations to scramble at step 0 if used with standard initialization. Superseded by `depthwise_conv_head` and `logit_boundary_head`.
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

### 6. Depthwise Convolutional Boundary Head with ReZero (`depthwise_conv_head`)
* **File**: [`predictive_heads/depthwise_conv.py`](./predictive_heads/depthwise_conv.py)
* **Papers**:
  - *ReZero is All You Need: Fast Convergence at Large Depth* (Bachlechner et al., 2021) — [https://arxiv.org/abs/2003.04887](https://arxiv.org/abs/2003.04887)
  - *Xception: Deep Learning with Depthwise Separable Convolutions* (Chollet, 2017) — [https://arxiv.org/abs/1610.02357](https://arxiv.org/abs/1610.02357)
  - *QANet: Combining Local Convolution with Global Self-Attention for Reading Comprehension* (Yu et al., 2018) — [https://arxiv.org/abs/1804.09541](https://arxiv.org/abs/1804.09541)
* **Mechanism**:
  1. Uses **depthwise convolution** (`groups=hidden_size`, $k=3$, padding $=1$), convolving each of the 768 feature dimensions independently across neighboring tokens. This reduces parameters from **1,769,472 to 2,304** ($99.87\%$ reduction), eliminating cross-channel scrambling.
  2. Implements **ReZero gating**:
     $$\tilde{H} = \text{LayerNorm}(H + \alpha \cdot \text{Dropout}(\text{GELU}(\text{DepthwiseConv}(H))))$$
     where $\alpha$ is a learnable scalar initialized to $0.0$.
* **Why it helps**:
  At Step 0, $\alpha = 0 \implies \tilde{H} = \text{LayerNorm}(H)$, which preserves $100\%$ of BERT's pretrained representations. The model safely learns local n-gram boundary filters without gradient shock.
* **Usage**:
  ```bash
  python train.py --predictive_head depthwise_conv_head
  ```

---

### 7. Logit-Space Boundary Smoothing Head (`logit_boundary_head`)
* **File**: [`predictive_heads/logit_boundary.py`](./predictive_heads/logit_boundary.py)
* **Papers**:
  - *Conditional Random Fields as Recurrent Neural Networks* (Zheng et al., 2015) — [https://arxiv.org/abs/1502.03240](https://arxiv.org/abs/1502.03240)
  - *Neural Architectures for Named Entity Recognition* (Lample et al., 2016) — [https://arxiv.org/abs/1603.01360](https://arxiv.org/abs/1603.01360)
* **Mechanism**:
  Instead of convolving high-dimensional 768-D token embeddings, this head operates strictly in **2D logit space**:
  1. Base logits are calculated from the proven linear projection:
     $$Z = H W + b \quad \in \mathbb{R}^{B \times L \times 2}$$
  2. A zero-initialized 1D convolution ($2 \to 2$ channels, $k=3$, padding $=1$) filters the raw logits:
     $$\Delta Z = \text{Conv1D}_{2 \to 2}(Z^T)^T \quad \text{(only 12 parameters!)}$$
  3. Final logits:
     $$\text{logits} = Z + \Delta Z$$
* **Why it helps**:
  Acts as a lightweight, differentiable local Markov Random Field / CRF. It models transition probabilities between adjacent token boundaries directly on the prediction scores without touching or risking damage to BERT's internal latent space.
* **Usage**:
  ```bash
  python train.py --predictive_head logit_boundary_head
  ```

---

## Running with SLURM Cluster Batch Script

To evaluate any head end-to-end on the compute cluster via [`train_eval.sh`](./train_eval.sh):

```bash
# Example: Depthwise Conv with ReZero
sbatch train_eval.sh --predictive_head depthwise_conv_head --learning_rate 3e-5 --head_learning_rate 1.5e-4 --epochs 2

# Example: Logit-Space Boundary Smoothing
sbatch train_eval.sh --predictive_head logit_boundary_head --learning_rate 3e-5 --head_learning_rate 1.5e-4 --epochs 2

# Example: Multi-Sample Dropout
sbatch train_eval.sh --predictive_head multi_sample_dropout_head --epochs 2

# Example: Conditioned Span Head
sbatch train_eval.sh --predictive_head conditioned_span_head --learning_rate 3e-5 --head_learning_rate 1.5e-4 --epochs 2
```

The model checkpoint, tokenizer, hyperparameters (`training_args.json`), predicted spans (`predictions.json`), and evaluation metrics (`eval_results.json`) will all be automatically stored in `checkpoints/<job_id>/`.
