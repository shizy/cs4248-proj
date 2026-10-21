import argparse
import json
import os
import random

from predictive_heads import PREDICTIVE_HEADS

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for Question Answering on SQuAD")

    # Model & Data paths
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased",
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--train_json_path", type=str, default="./data/train-v1.1.json",
                        help="Path to the training data JSON file.")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/bert_squad_finetune",
                        help="Directory to save checkpoint weights, tokenizer, and config.")
    parser.add_argument("--predictive_head", type=str, default="baseline_linear_head",
                        choices=list(PREDICTIVE_HEADS.keys()),
                        help="Prediction head architecture for span extraction.")

    # Tokenization & Dataset
    parser.add_argument("--max_seq_len", type=int, default=384,
                        help="Max total input sequence length after tokenization.")
    parser.add_argument("--doc_stride", type=int, default=128,
                        help="Stride length when splitting up long documents into chunks.")

    # Training dynamics
    parser.add_argument("-e", "--epochs", type=int, default=2,
                        help="Total number of training epochs to perform.")
    parser.add_argument("-b", "--batch_size", type=int, default=12,
                        help="Batch size per device for training.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-5,
                        help="Initial learning rate for AdamW (BERT backbone).")
    parser.add_argument("--head_learning_rate", type=float, default=None,
                        help="Initial learning rate for predictive head. If not set, defaults to --learning_rate.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for AdamW optimizer.")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of total training steps used for linear warmup.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for gradient clipping (<= 0 to disable).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization and shuffling.")

    # Logging and debugging
    parser.add_argument("--logging_steps", type=int, default=500,
                        help="Log training loss every N update steps.")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0: set total number of training steps to perform (overrides epochs). Useful for quick tests.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA even if available.")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, BertTokenizerFast, get_linear_schedule_with_warmup
    from bert import BertForQuestionAnswering
    from dataset import parse_squad_json, SquadDataset

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load tokenizer and model
    print(f"Loading tokenizer & model: {args.model_name_or_path}...")
    config = AutoConfig.from_pretrained(args.model_name_or_path) 
    config.head_type = args.predictive_head
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name_or_path)
    model = BertForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        config=config
    ).to(device)

    # Load dataset (TODO: move this out to a separate script so we can store to disk and not have to do this once per train)
    print("Loading dataset...")
    train_rows = parse_squad_json(args.train_json_path)
    train_dataset = SquadDataset(
        train_rows,
        tokenizer,
        max_length=args.max_seq_len,
        doc_stride=args.doc_stride,
        is_train=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Set optimizer and LR scheduler
    print("Initialising optimizer and scheduler...")
    head_lr = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
    print(f"Learning rates -> Backbone: {args.learning_rate} | Head: {head_lr}")

    optimizer_grouped_parameters = [
        {"params": [p for p in model.bert.parameters() if p.requires_grad], "lr": args.learning_rate},
        {"params": [p for p in model.pred_head.parameters() if p.requires_grad], "lr": head_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    total_steps = len(train_dataloader) * args.epochs if args.max_steps <= 0 else args.max_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Train
    print(f"Starting training for {args.epochs} epoch(s) ({total_steps} total steps)...")
    model.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )

            loss = outputs["loss"]
            epoch_loss += loss.item()

            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            global_step += 1

            if (step + 1) % args.logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                print(f"Epoch {epoch + 1}/{args.epochs} | Step {step + 1}/{len(train_dataloader)} | Avg Loss: {avg_loss:.4f}")

            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"Reached max_steps ({args.max_steps}). Stopping training early.")
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training arguments for reproducibility
    args_file = os.path.join(args.output_dir, "training_args.json")
    with open(args_file, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training complete. Model and config saved to {args.output_dir}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
