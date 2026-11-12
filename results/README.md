# Experiment Results

This directory catalogs evaluation outputs, model predictions, training artifacts, and statistical significance tests across all experimental tracks for SQuAD v1.1 question answering.

---

## Directory Overview

```text
results/
├── base/                # Backbone architecture & pre-training comparisons (BERT vs RoBERTa, TriviaQA)
├── predictive_heads/    # Prediction head architecture experiments & paired bootstrap test results
├── hidden_pooling/      # Encoder hidden layer aggregation strategies (last, mean, weighted, tokenwise)
└── best/                # Final top-performing model artifacts (RoBERTa-large + Conditioned Span Head)
```

---

## Subdirectories

### 1. `base/`
Contains evaluations of foundational backbone model variations and pre-training datasets:
* **Model scales & families:** `bert-large`, `roberta-base`, `roberta-large`.
* **Out-of-domain pre-training transfer:** Comparisons with TriviaQA pre-trained checkpoints (`bert_base_triviaqa`, `roberta_large_triviaqa`).
* Contains paired bootstrap statistical significance tests against baseline configurations.

### 2. `predictive_heads/`
Investigates span extraction prediction head architectures:
* **Architectures:** Baseline Linear, Multi-Sample Dropout, Residual MLP, Conditioned Span, and Logit-Space Boundary Smoothing.
* **Ablations:** Uniform learning rates vs. differentiated head learning rates (`--head_learning_rate`), training epochs (2 vs. 3), and cased vs. uncased backbones.
* See [**`results/predictive_heads/README.md`**](./predictive_heads/README.md) for detailed performance tables, metric logs, and paired bootstrap significance p-values.

### 3. `hidden_pooling/`
Explores aggregation of intermediate transformer layer representations:
* `baseline_last`: Standard top-layer hidden states.
* `mean`: Uniform mean pooling across all hidden layers.
* `weighted`: Learnable scalar layer-weighting across all layers.
* `tokenwise_weighted`: Token-level attention-weighted pooling across layers.

### 4. `best/`
Contains predictions, configurations, tokenizers, and evaluation results for the overall highest-performing model in this project:
* **Architecture:** `roberta-large` backbone with `conditioned_span_head`.
* **Performance:** **88.44% Exact Match (EM)** and **94.08% F1**.

---

## Common Files per Experiment

Each individual experiment run folder generally contains:
* **`eval_results.json`**: Standard SQuAD evaluation metrics (`exact`, `f1`, `total`, `HasAns_*`).
* **`predictions.json`**: Model answer span predictions mapped by question ID (`qid -> text`), used for evaluation and significance testing.
* **`training_args.json`**: CLI hyperparameters and flags used during training (learning rates, batch size, epochs, head type, pooler type, seed).
* **`training_metrics.json`**: Detailed step-by-step training logs (loss, grad norm, step evaluations).
* **`config.json`**: Hugging Face model architecture and configuration metadata.
* **`bootstrap_results.json` / `.csv`**: (Where applicable) 1,000-iteration paired bootstrap significance test results comparing against the baseline.
