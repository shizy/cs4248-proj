"""
Paired Bootstrap Testing for SQuAD Question Answering Models.

Performs non-parametric paired bootstrap significance testing between a test model
and a reference baseline model on Exact Match (EM) and F1 metrics.

Outputs:
  - bootstrap_results.csv : EM and F1 scores of both models across all bootstrap draws.
  - bootstrap_results.json: Summary containing bootstrap_config, observed_metrics,
                            and p_values (num_sig / total).
"""

import argparse
import collections
import csv
import json
import os
import re
import string
from typing import Optional, Any

import numpy as np


# ---------------------------------------------------------------------------
# SQuAD Evaluation Normalization & Scoring (Official SQuAD Evaluation Logic)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> float:
    return float(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def score_prediction(gold_answers: list[str], pred: str) -> tuple[float, float]:
    """Compute maximum EM and F1 score against all valid gold answers."""
    max_em = max(compute_exact(gold, pred) for gold in gold_answers)
    max_f1 = max(compute_f1(gold, pred) for gold in gold_answers)
    return max_em, max_f1


# ---------------------------------------------------------------------------
# Data Loading & Resolution Utilities
# ---------------------------------------------------------------------------

def resolve_data_path(data_path: Optional[str] = None) -> str:
    """Resolve gold SQuAD dataset path, defaulting to data/dev-v1.1.json."""
    if data_path and os.path.exists(data_path):
        return os.path.abspath(data_path)

    candidates = [
        data_path,
        "data/dev-v1.1.json",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dev-v1.1.json"),
        os.path.join(os.getcwd(), "data", "dev-v1.1.json"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)

    raise FileNotFoundError(
        "Could not find SQuAD evaluation dataset (e.g. data/dev-v1.1.json). "
        "Please ensure 'data/dev-v1.1.json' exists or pass data_path explicitly."
    )


def load_gold_data(data_path: str) -> tuple[list[str], dict[str, list[str]]]:
    """Load SQuAD JSON dataset and extract question IDs and normalized answers."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    articles = data_json.get("data", [])
    qids = []
    golds = {}

    for article in articles:
        for paragraph in article.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                qid = qa["id"]
                answers = [
                    a["text"] for a in qa.get("answers", [])
                    if normalize_answer(a.get("text", ""))
                ]
                if not answers:
                    answers = [""]  # For unanswerable questions
                qids.append(qid)
                golds[qid] = answers

    return qids, golds


def load_predictions(pred_path: str) -> tuple[str, dict[str, str]]:
    """
    Resolve and load prediction dictionary from file path or experiment directory.
    Returns (resolved_file_path, predictions_dict).
    """
    if os.path.isdir(pred_path):
        candidate = os.path.join(pred_path, "predictions.json")
        if os.path.exists(candidate):
            pred_path = candidate
        else:
            raise FileNotFoundError(f"predictions.json not found inside directory: {pred_path}")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    if not isinstance(preds, dict):
        raise ValueError(f"Expected JSON dictionary in {pred_path}, got {type(preds)}")

    return pred_path, preds


# ---------------------------------------------------------------------------
# Core Paired Bootstrap Testing Function
# ---------------------------------------------------------------------------

def paired_bootstrap_test(
    test_pred_path: str,
    ref_pred_path: str,
    data_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    num_samples: int = 1000,
    seed: Optional[int] = 42,
    output_csv_name: str = "bootstrap_results.csv",
    output_json_name: str = "bootstrap_results.json",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run paired bootstrap hypothesis testing between a test model and reference model.

    Args:
        test_pred_path: Path to the predictions JSON of the model to be tested.
        ref_pred_path: Path to the predictions JSON of the reference model.
        data_path: Path to gold SQuAD JSON dataset (default: auto-resolved to data/dev-v1.1.json).
        sample_size: Number of items drawn per bootstrap sample (default: size of test set).
        num_samples: Total number of bootstrap sample sets generated (default: 1000).
        seed: Random seed for reproducibility.
        output_csv_name: Filename of output CSV with per-sample metrics.
        output_json_name: Filename of output JSON with bootstrap_config, observed_metrics, p_values.
        verbose: Whether to print progress and summaries to stdout.

    Returns:
        dictionary containing bootstrap_config, observed_metrics, and p_values.
    """
    # 1. Resolve paths and load files
    resolved_test_path, test_preds = load_predictions(test_pred_path)
    resolved_ref_path, ref_preds = load_predictions(ref_pred_path)
    resolved_data_path = resolve_data_path(data_path)
    qids, golds = load_gold_data(resolved_data_path)

    total_questions = len(qids)
    if total_questions == 0:
        raise ValueError(f"No questions found in data file: {resolved_data_path}")

    # Output folder: folder containing the tested model's predictions
    output_folder = os.path.dirname(os.path.abspath(resolved_test_path))

    # Validate predictions coverage
    missing_test = [qid for qid in qids if qid not in test_preds]
    missing_ref = [qid for qid in qids if qid not in ref_preds]

    if missing_test:
        raise KeyError(
            f"Test predictions missing {len(missing_test)} questions (e.g. {missing_test[:3]})"
        )
    if missing_ref:
        raise KeyError(
            f"Reference predictions missing {len(missing_ref)} questions (e.g. {missing_ref[:3]})"
        )

    # 2. Precompute per-question EM and F1 scores for both models
    if verbose:
        print(f"Precomputing instance-level scores for {total_questions} questions...")

    test_em_scores = []
    test_f1_scores = []
    ref_em_scores = []
    ref_f1_scores = []

    for qid in qids:
        gold_answers = golds[qid]
        t_em, t_f1 = score_prediction(gold_answers, test_preds[qid])
        r_em, r_f1 = score_prediction(gold_answers, ref_preds[qid])

        test_em_scores.append(t_em)
        test_f1_scores.append(t_f1)
        ref_em_scores.append(r_em)
        ref_f1_scores.append(r_f1)

    # 3. Overall observed scores on the full test set
    obs_test_em = (sum(test_em_scores) / total_questions) * 100.0
    obs_test_f1 = (sum(test_f1_scores) / total_questions) * 100.0
    obs_ref_em = (sum(ref_em_scores) / total_questions) * 100.0
    obs_ref_f1 = (sum(ref_f1_scores) / total_questions) * 100.0

    obs_diff_em = obs_test_em - obs_ref_em
    obs_diff_f1 = obs_test_f1 - obs_ref_f1

    # Effective sample size per bootstrap draw
    effective_sample_size = sample_size if (sample_size is not None and sample_size > 0) else total_questions

    if verbose:
        print("=================================================================")
        print("PAIRED BOOTSTRAP TEST CONFIGURATION")
        print(f"  Test Model Predictions : {resolved_test_path}")
        print(f"  Ref Model Predictions  : {resolved_ref_path}")
        print(f"  Total Questions (N)    : {total_questions}")
        print(f"  Sample Size (M)        : {effective_sample_size}")
        print(f"  Number of Samples (B)  : {num_samples}")
        print(f"  Random Seed            : {seed}")
        print("-----------------------------------------------------------------")
        print(f"  Observed Test Model    : EM = {obs_test_em:.2f}%, F1 = {obs_test_f1:.2f}%")
        print(f"  Observed Ref Model     : EM = {obs_ref_em:.2f}%, F1 = {obs_ref_f1:.2f}%")
        print(f"  Observed Difference (T - R) : Diff EM = {obs_diff_em:+.2f}%, Diff F1 = {obs_diff_f1:+.2f}%")
        print("=================================================================")

    # 4. Generate Bootstrap Samples using Vectorized NumPy
    test_em_arr = np.array(test_em_scores, dtype=np.float64)
    test_f1_arr = np.array(test_f1_scores, dtype=np.float64)
    ref_em_arr = np.array(ref_em_scores, dtype=np.float64)
    ref_f1_arr = np.array(ref_f1_scores, dtype=np.float64)

    rng = np.random.default_rng(seed)
    # Shape: (num_samples, effective_sample_size)
    sample_indices = rng.integers(0, total_questions, size=(num_samples, effective_sample_size))

    boot_test_em_arr = test_em_arr[sample_indices].mean(axis=1) * 100.0
    boot_test_f1_arr = test_f1_arr[sample_indices].mean(axis=1) * 100.0
    boot_ref_em_arr = ref_em_arr[sample_indices].mean(axis=1) * 100.0
    boot_ref_f1_arr = ref_f1_arr[sample_indices].mean(axis=1) * 100.0

    boot_diff_em_arr = boot_test_em_arr - boot_ref_em_arr
    boot_diff_f1_arr = boot_test_f1_arr - boot_ref_f1_arr

    bootstrap_rows = []
    for b in range(num_samples):
        bootstrap_rows.append({
            "sample_id": b + 1,
            "model_em": round(float(boot_test_em_arr[b]), 4),
            "model_f1": round(float(boot_test_f1_arr[b]), 4),
            "ref_em": round(float(boot_ref_em_arr[b]), 4),
            "ref_f1": round(float(boot_ref_f1_arr[b]), 4),
            "diff_em": round(float(boot_diff_em_arr[b]), 4),
            "diff_f1": round(float(boot_diff_f1_arr[b]), 4),
        })

    # 5. Compute P-Values following Lecture Slides algorithm:
    # H0: delta(x) <= 0 (A is not better than B)
    # H1: delta(x) > 0  (A is better than B)
    # For i = 1 to b:
    #   s = s + 1 if delta(x_i) <= 0
    # p-value = s / b
    num_sig_em = int(np.sum(boot_diff_em_arr <= 0))
    p_val_em = num_sig_em / float(num_samples)

    num_sig_f1 = int(np.sum(boot_diff_f1_arr <= 0))
    p_val_f1 = num_sig_f1 / float(num_samples)

    # 6. Write CSV Output
    output_csv_path = os.path.join(output_folder, output_csv_name)
    csv_fieldnames = ["sample_id", "model_em", "model_f1", "ref_em", "ref_f1", "diff_em", "diff_f1"]

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(bootstrap_rows)

    # 7. Write JSON Output (bootstrap_config, observed_metrics, p_values only)
    output_json_path = os.path.join(output_folder, output_json_name)
    results_summary = {
        "bootstrap_config": {
            "test_predictions_file": os.path.basename(resolved_test_path),
            "reference_predictions_file": os.path.basename(resolved_ref_path),
            "test_predictions_path": resolved_test_path,
            "reference_predictions_path": resolved_ref_path,
            "dataset_size": total_questions,
            "sample_size": effective_sample_size,
            "num_samples": num_samples,
            "seed": seed,
        },
        "observed_metrics": {
            "test_model": {
                "exact_match": round(obs_test_em, 4),
                "f1_score": round(obs_test_f1, 4),
            },
            "reference_model": {
                "exact_match": round(obs_ref_em, 4),
                "f1_score": round(obs_ref_f1, 4),
            },
            "observed_difference": {
                "delta_exact_match": round(obs_diff_em, 4),
                "delta_f1_score": round(obs_diff_f1, 4),
            },
        },
        "p_values": {
            "exact_match": {
                "p_value": round(float(p_val_em), 4),
                "num_sig": int(num_sig_em),
                "total": int(num_samples),
            },
            "f1_score": {
                "p_value": round(float(p_val_f1), 4),
                "num_sig": int(num_sig_f1),
                "total": int(num_samples),
            },
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as jsonfile:
        json.dump(results_summary, jsonfile, indent=2)

    if verbose:
        print("BOOTSTRAP RESULTS SUMMARY:")
        print(f"  Exact Match Diff (Test - Ref) : {obs_diff_em:+.4f}%")
        print(f"    p-value: {p_val_em:.4f} (num_sig: {num_sig_em}/{num_samples})")
        print(f"  F1 Score Diff (Test - Ref)    : {obs_diff_f1:+.4f}%")
        print(f"    p-value: {p_val_f1:.4f} (num_sig: {num_sig_f1}/{num_samples})")
        print(f"  Results saved to directory    : {output_folder}")
        print(f"    - CSV samples : {output_csv_path}")
        print(f"    - JSON summary: {output_json_path}")
        print("=================================================================")

    return results_summary


# Programmatic alias
run_paired_bootstrap_test = paired_bootstrap_test


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run paired bootstrap significance testing between two SQuAD model predictions."
    )
    parser.add_argument(
        "--test_pred_path", "-t", "--test_pred",
        type=str,
        required=True,
        help="Path to the predictions JSON file (or experiment directory) of the model to be tested.",
    )
    parser.add_argument(
        "--ref_pred_path", "-r", "--ref_pred",
        type=str,
        required=True,
        help="Path to the predictions JSON file (or experiment directory) of the reference model.",
    )
    parser.add_argument(
        "--data_path", "-d", "--data",
        type=str,
        default=None,
        help="Path to SQuAD test/dev JSON dataset (optional, defaults to data/dev-v1.1.json).",
    )
    parser.add_argument(
        "--sample_size", "-s",
        type=int,
        default=None,
        help="Number of questions per bootstrap sample (default: size of test set).",
    )
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=1000,
        help="Total number of bootstrap samples to draw (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42). Set to -1 to randomize.",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="bootstrap_results.csv",
        help="Name of output CSV file to write inside test model folder (default: bootstrap_results.csv).",
    )
    parser.add_argument(
        "--json_name",
        type=str,
        default="bootstrap_results.json",
        help="Name of output JSON file to write inside test model folder (default: bootstrap_results.json).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed = None if args.seed == -1 else args.seed

    paired_bootstrap_test(
        test_pred_path=args.test_pred_path,
        ref_pred_path=args.ref_pred_path,
        data_path=args.data_path,
        sample_size=args.sample_size,
        num_samples=args.num_samples,
        seed=seed,
        output_csv_name=args.csv_name,
        output_json_name=args.json_name,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
