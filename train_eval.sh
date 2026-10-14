#!/bin/bash
#SBATCH --job-name=train_eval
#SBATCH --output=output-%j.out
#SBATCH --error=output-%j.out
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=h100-96

#SBATCH --mail-type=BEGIN,END,FAIL
# Import set your email to recieve alert of job completion / failure
#SBATCH --mail-user=your_email@example.com

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Force unbuffered standard output/error so Python flushes logs immediately into the .out file
export PYTHONUNBUFFERED=1

# Resolve Job ID (fallback to timestamp if run outside SLURM)
JOB_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="checkpoints/${JOB_ID}"
DATA_DIR="./data"
TRAIN_FILE="${DATA_DIR}/train-v1.1.json"
DEV_FILE="${DATA_DIR}/dev-v1.1.json"
PRED_FILE="${CHECKPOINT_DIR}/predictions.json"
EVAL_RESULT_FILE="${CHECKPOINT_DIR}/eval_results.json"

# Create checkpoint output directory
mkdir -p "${CHECKPOINT_DIR}"

echo "=========================================================="
echo "Job ID             : ${JOB_ID}"
echo "Node               : $(hostname)"
echo "Start Time         : $(date)"
echo "Checkpoint Dir     : ${CHECKPOINT_DIR}"
echo "Log File           : output-${JOB_ID}.out"
echo "=========================================================="

# Activate virtual environment
source ./.venv/bin/activate
pip install -r requirements.txt

# ---------------------------------------------------------
# Step 1: Train Model
# ---------------------------------------------------------
echo ""
echo ">>> [Step 1/3] Starting Training..."
python -u train.py \
    --output_dir "${CHECKPOINT_DIR}" \
    --train_json_path "${TRAIN_FILE}" \
    "$@"

# ---------------------------------------------------------
# Step 2: Generate Predictions on Validation Set
# ---------------------------------------------------------
echo ""
echo ">>> [Step 2/3] Generating Predictions..."
python -u predict.py \
    --model_path "${CHECKPOINT_DIR}" \
    --val_json_path "${DEV_FILE}" \
    --output_pred_file "${PRED_FILE}"

# ---------------------------------------------------------
# Step 3: Evaluate Predictions
# ---------------------------------------------------------
echo ""
echo ">>> [Step 3/3] Evaluating Model Predictions..."
python -u evaluate-v2.0.py \
    "${DEV_FILE}" \
    "${PRED_FILE}" \
    --out-file "${EVAL_RESULT_FILE}"

echo ""
echo "=========================================================="
echo "Final Evaluation Metrics (${JOB_ID}):"
cat "${EVAL_RESULT_FILE}"
echo ""
echo "Job completed successfully at: $(date)"
echo "=========================================================="

