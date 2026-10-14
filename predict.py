import argparse
import collections
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast
from bert import BertForQuestionAnswering
from dataset import parse_squad_json, SquadDataset

def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "token_type_ids": torch.stack([x["token_type_ids"] for x in batch]),
        "offset_mapping": [x["offset_mapping"] for x in batch],
        "sequence_ids": [x["sequence_ids"] for x in batch],
        "q_id": [x["q_id"] for x in batch],
        "context": [x["context"] for x in batch],
    }

# ---------------------------------------------------------
# Inference & Span Selection
# ---------------------------------------------------------
def generate_predictions(
    model_path,
    val_json_path,
    output_pred_file="predictions.json",
    batch_size=16,
    n_best_size=20,
    max_answer_length=30,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {model_path} onto {device}")

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = BertForQuestionAnswering.from_pretrained(model_path).to(device)
    model.eval()

    raw_val = parse_squad_json(val_json_path)
    val_dataset = SquadDataset(raw_val, tokenizer, is_train=False)
    dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Accumulate candidate spans per question id
    qid_to_candidates = collections.defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]

            start_logits = start_logits.cpu().tolist()
            end_logits = end_logits.cpu().tolist()

            for i in range(len(batch["q_id"])):
                q_id = batch["q_id"][i]
                context = batch["context"][i]
                offsets = batch["offset_mapping"][i]
                seq_ids = batch["sequence_ids"][i]
                s_logits = start_logits[i]
                e_logits = end_logits[i]

                # Top-N indices (For Top-N candidate search, as seen from google-research/bert/run_squad.py)
                top_start = sorted(range(len(s_logits)), key=lambda k: s_logits[k], reverse=True)[:n_best_size]
                top_end = sorted(range(len(e_logits)), key=lambda k: e_logits[k], reverse=True)[:n_best_size]

                for s_idx in top_start:
                    for e_idx in top_end:
                        # Validate boundaries: sequence_id must be 1 (context)
                        if (
                            s_idx >= len(offsets)
                            or e_idx >= len(offsets)
                            or seq_ids[s_idx] != 1
                            or seq_ids[e_idx] != 1
                            or e_idx < s_idx
                            or (e_idx - s_idx + 1) > max_answer_length
                        ):
                            continue

                        char_start = offsets[s_idx][0]
                        char_end = offsets[e_idx][1]
                        pred_text = context[char_start:char_end]
                        score = s_logits[s_idx] + e_logits[e_idx]

                        qid_to_candidates[q_id].append({"score": score, "text": pred_text})

    # Pick the best scoring span per question ID
    final_preds = {}
    for row in raw_val:
        q_id = row["id"]
        candidates = qid_to_candidates.get(q_id, [])
        if not candidates:
            final_preds[q_id] = ""
        else:
            best_candidate = max(candidates, key=lambda x: x["score"])
            final_preds[q_id] = best_candidate["text"]

    out_dir = os.path.dirname(output_pred_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_pred_file, "w", encoding="utf-8") as f:
        json.dump(final_preds, f, indent=2)

    print(f"Predictions written to {output_pred_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions using fine-tuned BERT model on SQuAD")
    parser.add_argument("--model_path", type=str, default="./checkpoints/bert_squad_finetune",
                        help="Path to fine-tuned model checkpoint directory.")
    parser.add_argument("--val_json_path", type=str, default="./data/dev-v1.1.json",
                        help="Path to validation/dev SQuAD JSON file.")
    parser.add_argument("--output_pred_file", type=str, default="./data/predictions.json",
                        help="Path to write output predictions JSON.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for prediction DataLoader.")
    parser.add_argument("--n_best_size", type=int, default=20,
                        help="Total number of n-best predictions to generate per question.")
    parser.add_argument("--max_answer_length", type=int, default=30,
                        help="Maximum length of an answer that can be generated.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_predictions(
        model_path=args.model_path,
        val_json_path=args.val_json_path,
        output_pred_file=args.output_pred_file,
        batch_size=args.batch_size,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
    )
