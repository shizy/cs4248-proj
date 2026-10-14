# Instructions
- Ensure that `torch` and `transformers==4.56.1` are installed (preferably within a virtual environment. You can `pip install -r requirements.txt` in a fresh virtual environment to get the exact libraries being used; see [`requirements.txt`](./requirements.txt)).
- Download the `dev-v1.1.json` and `train-v1.1.json` files from Canvas into the `./data` directory.
- Run [`train.py`](./train.py) to train the model. Hyperparameters can be configured via CLI flags (run `python train.py --help` for available options). After completing training, the checkpoint and `training_args.json` are saved locally.
  - Run `python train.py --help` to see the list of configurable params.
- Run [`predict.py`](./predict.py), ensuring that `--model_path` and `--val_json_path` are correct. This goes over the dev dataset and generates QA spans for each example with our fine-tuned model, along with the respective question ID, as a JSON file. This file is used by [`evaluate-v2.0.py`](./evaluate-v2.0.py) for evals.
- Run [`evaluate-v2.0.py`](./evaluate-v2.0.py), with the dev dataset path and predictions JSON path. This prints out the evaluation metric scores.

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

# File Reference

| File | Description |
| :--- | :--- |
| [`train.py`](./train.py) | Main fine-tuning script with CLI argument parsing via `argparse`, linear LR warmup scheduling, and checkpoint saving. |
| [`predict.py`](./predict.py) | Inference script that evaluates fine-tuned checkpoints on validation passages to generate question-answering span predictions. |
| [`evaluate-v2.0.py`](./evaluate-v2.0.py) | Official SQuAD evaluation script computing Exact Match (EM) and F1 metric scores against ground truth answers. |
| [`bert.py`](./bert.py) | Custom PyTorch model implementation (`BertForQuestionAnswering`) that delegates span extraction to the selected predictive head. |
| [`dataset.py`](./dataset.py) | SQuAD dataset parser (`parse_squad_json`) and PyTorch `SquadDataset` handling tokenization and sliding window context chunking. |
| [`predictive_heads/head.py`](./predictive_heads/head.py) | Abstract base class (`PredictiveHead`), `@register_head` decorator, and `get_pred_head` factory. |
| [`predictive_heads/baseline.py`](./predictive_heads/baseline.py) | Baseline linear predictive head (`baseline_linear_head`). |
| [`predictive_heads/__init__.py`](./predictive_heads/__init__.py) | Auto-discovery package loader that dynamically registers all head modules in the directory. |
| [`train_eval.sh`](./train_eval.sh) | Automated SLURM batch job script running end-to-end training, prediction, and evaluation on cluster nodes with unbuffered logging. |
| [`train.sh`](./train.sh) | Base SLURM batch script for running training. |
| [`explore_data.py`](./explore_data.py) | Utility script to inspect the nested structure of SQuAD JSON datasets. |
| [`requirements.txt`](./requirements.txt) | Environment package dependencies and library versions. |

# Todos
1. Add argparse into the train script to allow for hyperparameter/arch changes via the command line (done for training hyperparams)
2. Abstract out the linear layer in the model to make it extensible. Make it modifiable in the training args (done via --predictive_head and predictive_heads/)
4. Support more interesting LR scheduling algorithms. Abstract that out to be modifable in the training args (not really important)
5. Support more interesting losses. Abstract that out to be modifiable in the training args