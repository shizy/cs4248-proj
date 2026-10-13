# Instructions
- Ensure that `torch` and `transformers==4.56.1` are installed (preferably within a virtual environment. You can `pip install -r requirements.txt` in a fresh virtual environment to get the exact libraries being used)
- Download the `dev-v1.1.json` and `train-v1.1.json` files from canvas into the `./data` directory
- Run `train.py` (will be improved on). Currently hyperparameters are hardcoded in. Ensure that `train_json_path` is correct. After completing training, the checkpoint will be saved locally
- Run `predict.py`, ensuring that `model_path` and `val_json_path` are correct. This goes over the dev dataset and generates the QA spans for each example with our finetuned model, along with the respective question ID, as a json file. This json file is used by `evaluate-v2.0.py` for evals
- Run `evaluate-v2.0.py`, with the dev dataset path and predictions json path. This prints out the metric scores

# SoC Compute Cluster Guide
Due to the amount of data points, it is ideal to use the SoC Compute Cluster to run both training (`train.py`) and inference (`predict.py`)

In the remote environment, we submit jobs via slurm. We can do `srun` to execute an interactive job, or `sbatch` to execute a batch job.

When I execute an interactive job, I run something like this:
```
srun --mem=32G --gpus=h100-96 --time=02:00:00 bash -c "source ./.venv/bin/activate; python -u train.py"
```

This submits a job for the node with H100 96GB. On getting access to the node, it execs bash with instructions to activate the virtual environment on the GPU node and run `train.py` unbuffered (normal buffered stdout doesn't play nice with slurm srun)

For long jobs it's probably better to run it as a batch job (`sbatch`) and forget about it

# Todos
1. Add argparse into the train script to allow for hyperparameter/arch changes via the command line
2. Abstract out the linear layer in the model to make it extensible. Make it modifiable in the training args
4. Support more interesting LR scheduling algorithms. Abstract that out to be modifable in the training args (not really important)
5. Support more interesting losses. Abstract that out to be modifiable in the training args