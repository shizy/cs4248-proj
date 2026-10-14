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

# File Reference

| File | Description |
| :--- | :--- |
| [`train.py`](./train.py) | Main fine-tuning script with CLI argument parsing via `argparse`, linear LR warmup scheduling, and checkpoint saving. |
| [`predict.py`](./predict.py) | Inference script that evaluates fine-tuned checkpoints on validation passages to generate question-answering span predictions. |
| [`evaluate-v2.0.py`](./evaluate-v2.0.py) | Official SQuAD evaluation script computing Exact Match (EM) and F1 metric scores against ground truth answers. |
| [`bert.py`](./bert.py) | Custom PyTorch model implementation (`BertForQuestionAnswering`) with a classification head for span start/end logits. |
| [`dataset.py`](./dataset.py) | SQuAD dataset parser (`parse_squad_json`) and PyTorch `SquadDataset` handling tokenization and sliding window context chunking. |
| [`train_eval.sh`](./train_eval.sh) | Automated SLURM batch job script running end-to-end training, prediction, and evaluation on cluster nodes with unbuffered logging. |
| [`train.sh`](./train.sh) | Base SLURM batch script for running training. |
| [`explore_data.py`](./explore_data.py) | Utility script to inspect the nested structure of SQuAD JSON datasets. |
| [`requirements.txt`](./requirements.txt) | Environment package dependencies and library versions. |

# Todos
1. Add argparse into the train script to allow for hyperparameter/arch changes via the command line (done for training hyperpramams)
2. Abstract out the linear layer in the model to make it extensible. Make it modifiable in the training args
4. Support more interesting LR scheduling algorithms. Abstract that out to be modifable in the training args (not really important)
5. Support more interesting losses. Abstract that out to be modifiable in the training args