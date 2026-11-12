# Instructions
- Ensure that `torch` and `transformers==4.56.1` are installed (preferably within a virtual environment. You can `pip install -r requirements.txt` in a fresh virtual environment to get the exact libraries being used; see [`requirements.txt`](./requirements.txt)).
- Download the `dev-v1.1.json` and `train-v1.1.json` files from Canvas into the `./data` directory.
- Run [`train.py`](./train.py) to train the model. Hyperparameters can be configured via CLI flags (run `python train.py --help` for available options). After completing training, the checkpoint and `training_args.json` are saved locally.
  - Run `python train.py --help` to see the list of configurable params.
- Run [`predict.py`](./predict.py), ensuring that `--model_path` and `--val_json_path` are correct. This goes over the dev dataset and generates QA spans for each example with our fine-tuned model, along with the respective question ID, as a JSON file. This file is used by [`evaluate-v2.0.py`](./evaluate-v2.0.py) for evals.
- Run [`evaluate-v2.0.py`](./evaluate-v2.0.py), with the dev dataset path and predictions JSON path. This prints out the evaluation metric scores.
- Run [`paired_bootstrap_test.py`](./paired_bootstrap_test.py) to test statistical significance (EM & F1) between a test model checkpoint and a reference baseline model. Results (`bootstrap_results.csv` and `bootstrap_results.json`) are automatically saved into the tested model's directory.



# SoC Compute Cluster Guide
Due to the amount of data points, it is ideal to use the SoC Compute Cluster to run both training ([`train.py`](./train.py)) and inference ([`predict.py`](./predict.py))

In the remote environment, we submit jobs via slurm. We can do `srun` to execute an interactive job, or `sbatch` to execute a batch job.

When I execute an interactive job, I run something like this:
```
srun --mem=32G --gpus=h100-96 --time=02:00:00 bash -c "source ./.venv/bin/activate; python -u train.py"
```

This submits a job for the node with H100 96GB. On getting access to the node, it execs bash with instructions to activate the virtual environment on the GPU node and run `train.py` unbuffered (normal buffered stdout doesn't play nice with slurm srun)

For long jobs it's probably better to run it as a batch job (`sbatch`) and forget about it

A [`train_eval.sh`](./train_eval.sh) eval script is created to run training, prediction, then evaluation automatically using `sbatch`:
```
sbatch train_eval.sh
```
Set your nus email to receive email alerts from slurm `#SBATCH --mail-user=your_email@example.com`.

Additional training params can be amended in the script under the `Step 1: Train Model` section, for instance:
```
# ---------------------------------------------------------
# Step 1: Train Model
# ---------------------------------------------------------
echo ""
echo ">>> [Step 1/3] Starting Training..."
python -u train.py \
    --output_dir "${CHECKPOINT_DIR}" \
    --train_json_path "${TRAIN_FILE}" \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --warmup_ratio 0.1 \
    --seed 42 \
    "$@"
```

Or it can be directly injected to a `sbatch` run with `train_eval.sh`:
```
sbatch train_eval.sh --batch_size 32
```


# Using Other Backbones (BERT-large, RoBERTa)

The backbone is chosen with `--model_name_or_path`. BERT and RoBERTa checkpoints are supported (e.g. `bert-base-uncased`, `bert-large-uncased`, `roberta-base`, `roberta-large`); the right model class is picked from the checkpoint's config (see `get_qa_model_class` in [`bert.py`](./bert.py)), and [`predict.py`](./predict.py) reads it back from the saved `config.json`, so no extra flag is needed there. Other architectures raise an "Unsupported backbone" error.

Baseline (default linear head) runs of the larger backbones on the cluster. `sbatch` command-line options override the `#SBATCH` lines in the script, so give the large models a longer `--time`:
```
sbatch --time=04:00:00 --job-name=bert-large    train_eval.sh --model_name_or_path bert-large-uncased
sbatch                 --job-name=roberta-base  train_eval.sh --model_name_or_path roberta-base
sbatch --time=04:00:00 --job-name=roberta-large train_eval.sh --model_name_or_path roberta-large
```
Each job saves `checkpoints/<job_id>/eval_results.json` (EM / F1) next to `training_args.json` (records the backbone, head and hyperparameters used), and prints the metrics at the end of `output-<job_id>.out`. `checkpoints/` is git-ignored, so copy the numbers somewhere shared.

Note: RoBERTa-large can diverge with a high learning rate. The loss starts near 5.9 (≈ ln 384, i.e. random guessing); if `Avg Loss` is still around 5-6 after the first few thousand steps, the run is not learning, so stop it and rerun with a lower rate, e.g. `--learning_rate 1.5e-5`.


# Adding Custom Predictive Heads

The codebase uses a modular registry with auto-discovery located in the [`predictive_heads/`](./predictive_heads/) directory.

### How It Works
1. Any Python file placed inside [`predictive_heads/`](./predictive_heads/) is automatically imported by [`predictive_heads/__init__.py`](./predictive_heads/__init__.py).
2. The `@register_head("<name>")` decorator registers the class into the `PREDICTIVE_HEADS` registry.
3. The registered head is immediately exposed in [`train.py`](./train.py) under the `--predictive_head` CLI argument and validated in choices.
4. When saved via `model.save_pretrained()`, the chosen head type is stored inside `config.json` (`config.head_type`), so [`predict.py`](./predict.py) loads the correct head automatically during inference without manual flags.

### Step-by-Step Guide: Adding a New Head

1. Create a new file in `predictive_heads/` (e.g., `predictive_heads/mlp.py`).
2. Subclass [`PredictiveHead`](./predictive_heads/head.py) and decorate the class with `@register_head("<name>")`:

```python
import torch.nn as nn
from .head import PredictiveHead, register_head

@register_head("head_name")
class NewHead(PredictiveHead):
    def __init__(self, config)
        super().__init__(config)
        ...

    def forward(self, seq_output):
        ...
        return start_logits, end_logits
```

3. Train using your new head with the CLI flag:
```bash
python train.py --predictive_head mlp_head
```
or in your SLURM submission:
```bash
sbatch train_eval.sh --predictive_head mlp_head
```

# Adding Custom Pooling Methods

Similar to the predictive heads, the codebase contains a modular registry with auto-discovery for hidden state pooling, located in the [`hidden_pooling/`](./hidden_pooling/) directory. 

### How It Works

1. Any Python file placed inside [`hidden_pooling/`](./hidden_pooling/) is automatically imported by [`hidden_pooling/__init__.py`](./hidden_pooling/__init__.py).
2. The `@register_pooler("<name>")` decorator registers the class into the `HIDDEN_POOLERS` registry.
3. The registered pooler is immediately exposed in [`train.py`](./train.py) under the `--pooler_type` CLI argument and validated in choices.
4. When saved via `model.save_pretrained()`, the chosen pooler type is stored inside `config.json` (`config.head_type`), so [`predict.py`](./predict.py) loads the correct pooler automatically during inference without manual flags.

### Running Training with custom poolers

Train using a custom pooler with the CLI flag:
```bash
python train.py --pooler_type mean
```
or in your SLURM submission:
```bash
sbatch train_eval.sh --pooler_type mean
```

Note that wthout specifying `--pooler_type`, the training script defaults to last layer pooling.

# Running Paired Bootstrap Significance Testing

To verify whether a model's performance improvements over a reference baseline are statistically significant, use [`paired_bootstrap_test.py`](./paired_bootstrap_test.py).


### Hypothesis & Methodology

Following non-parametric paired bootstrap testing:
- $H_0: \delta(x) \le 0$ (Candidate Model $A$ is not better than Baseline Model $B$)
- $H_1: \delta(x) > 0$ (Candidate Model $A$ is better than Baseline Model $B$)

For each bootstrap iteration $i = 1 \dots b$ ($b = 1000$ by default):
1. Create a bootstrap test set $x_i$ of the same size as $x$ by sampling from $x$ with replacement.
2. Compute metric difference $\delta(x_i) = M(A, x_i) - M(B, x_i)$ for both Exact Match (EM) and F1.
3. Increment counter $s \leftarrow s + 1$ if $\delta(x_i) \le 0$.
4. Compute empirical $p\text{-value} = \frac{s}{b}$. If $p < 0.01$ (or $0.05$), reject $H_0$ and conclude Model $A$ is significantly better than Model $B$.

### CLI Usage

```bash
python paired_bootstrap_test.py \
    --test_pred_path results/predictive_heads/conditioned_span_cased_3ep_diff_lr/predictions.json \
    --ref_pred_path results/predictive_heads/baseline_linear/predictions.json \
    --num_samples 1000 \
    --seed 42
```

> **Note**: Both `--test_pred_path` and `--ref_pred_path` can point either directly to `predictions.json` or to the experiment directory containing it. `--data_path` is optional and automatically resolves to `data/dev-v1.1.json`.

Available arguments:
- `--test_pred_path`, `-t`: Path to predictions JSON (or folder) of the model being tested (**required**).
- `--ref_pred_path`, `-r`: Path to predictions JSON (or folder) of the reference baseline model (**required**).
- `--num_samples`, `-n`: Total number of bootstrap sample draws (default: `1000`).
- `--data_path`, `-d`: Path to gold SQuAD evaluation JSON (default: `data/dev-v1.1.json`).
- `--sample_size`, `-s`: Number of questions per draw (default: test set size, 10,570).
- `--seed`: Random seed for sampling reproducibility (default: `42`). Set to `-1` for random.
- `--quiet`, `-q`: Suppress console logging.

### Programmatic Python Usage

```python
from paired_bootstrap_test import paired_bootstrap_test

results = paired_bootstrap_test(
    test_pred_path="results/predictive_heads/conditioned_span_cased_3ep_diff_lr/predictions.json",
    ref_pred_path="results/predictive_heads/baseline_linear/predictions.json",
    num_samples=1000,
    seed=42,
)

print("Exact Match:", results["p_values"]["exact_match"])
# {'p_value': 0.19, 'num_sig': 190, 'total': 1000}

print("F1 Score:", results["p_values"]["f1_score"])
# {'p_value': 0.102, 'num_sig': 102, 'total': 1000}
```

### Outputs

Both outputs are automatically written directly into the folder of the tested model:
- `bootstrap_results.csv`: Contains per-draw metrics (`sample_id,model_em,model_f1,ref_em,ref_f1,diff_em,diff_f1`).
- `bootstrap_results.json`: Summary JSON containing `bootstrap_config`, `observed_metrics`, and `p_values` (`num_sig` / `total`).

# File Reference

| File | Description |
| :--- | :--- |
| [`train.py`](./train.py) | Main fine-tuning script with CLI argument parsing via `argparse`, linear LR warmup scheduling, and checkpoint saving. |
| [`predict.py`](./predict.py) | Inference script that evaluates fine-tuned checkpoints on validation passages to generate question-answering span predictions. |
| [`evaluate-v2.0.py`](./evaluate-v2.0.py) | Official SQuAD evaluation script computing Exact Match (EM) and F1 metric scores against ground truth answers. |
| [`bert.py`](./bert.py) | Custom PyTorch models (`BertForQuestionAnswering`, `RobertaForQuestionAnswering`, chosen via `get_qa_model_class`) that delegate span extraction to the selected predictive head. |
| [`paired_bootstrap_test.py`](./paired_bootstrap_test.py) | Non-parametric paired bootstrap test comparing a candidate model against a baseline for EM & F1 statistical significance ($p$-values, $s/b$, and sample CSVs). |
| [`dataset.py`](./dataset.py) | SQuAD dataset parser (`parse_squad_json`) and PyTorch `SquadDataset` handling tokenization and sliding window context chunking. |
| [`predictive_heads/head.py`](./predictive_heads/head.py) | Abstract base class (`PredictiveHead`), `@register_head` decorator, and `get_pred_head` factory. |
| [`predictive_heads/baseline.py`](./predictive_heads/baseline.py) | Baseline linear predictive head (`baseline_linear_head`). |
| [`predictive_heads/multi_sample_dropout.py`](./predictive_heads/multi_sample_dropout.py) | Multi-sample dropout head for accelerated convergence and variance reduction (`multi_sample_dropout_head`). |
| [`predictive_heads/residual_mlp.py`](./predictive_heads/residual_mlp.py) | Deep residual MLP head with LayerNorm and GELU non-linearity (`residual_mlp_head`). |
| [`predictive_heads/conv_boundary.py`](./predictive_heads/conv_boundary.py) | 1D Convolutional local n-gram boundary head (`conv_boundary_head`). |
| [`predictive_heads/conditioned_span.py`](./predictive_heads/conditioned_span.py) | Autoregressive span head conditioning end prediction on start token representation (`conditioned_span_head`). |
| [`predictive_heads/depthwise_conv.py`](./predictive_heads/depthwise_conv.py) | Depthwise 1D conv with ReZero gating for safe local boundary filtering (`depthwise_conv_head`). |
| [`predictive_heads/logit_boundary.py`](./predictive_heads/logit_boundary.py) | Differentiable logit-space boundary smoothing filter (`logit_boundary_head`). |
| [`predictive_heads/__init__.py`](./predictive_heads/__init__.py) | Auto-discovery package loader that dynamically registers all head modules in the directory. |
| [`train_eval.sh`](./train_eval.sh) | Automated SLURM batch job script running end-to-end training, prediction, and evaluation on cluster nodes with unbuffered logging. |
| [`train.sh`](./train.sh) | Base SLURM batch script for running training. |
| [`explore_data.py`](./explore_data.py) | Utility script to inspect the nested structure of SQuAD JSON datasets. |
| [`requirements.txt`](./requirements.txt) | Environment package dependencies and library versions. |
| [`results/predictive_heads/`](./results/predictive_heads/README.md) | Various results for predictive head experiments. |

