import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from bert import BertForQuestionAnswering
from dataset import parse_squad_json, SquadDataset

if __name__ == "__main__":
    train_json_path = "./data/train-v1.1.json"

    model_name = "bert-base-uncased"
    max_seq_len = 384
    doc_stride = 128
    batch_size = 12
    learning_rate = 3e-5
    epochs = 2
    warmup_ratio = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name).to(device)

    # Load dataset (TODO: move this out to a separate script so we can store to disk and not have to do this once per train)
    print("Loading dataset...")
    train_rows = parse_squad_json(train_json_path)
    train_dataset = SquadDataset(train_rows, tokenizer, max_seq_len, doc_stride, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set optimizer and LR scheduler
    print("Initialising optimizer and scheduler...")
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Train
    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0.0
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
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # ?

            optimizer.step()
            scheduler.step()

            # # Debug. To remove.
            # if step == 5:
            #     avg_loss = total_train_loss / step
            #     print(f"Step {step}/{len(train_dataloader)} | Avg Loss: {avg_loss}")
            #     break

            if (step + 1) % 500 == 0:
                avg_loss = total_train_loss / (step + 1)
                print(f"Step {step + 1}/{len(train_dataloader)} | Avg Loss: {avg_loss}")

    model.save_pretrained("./checkpoints/bert_squad_finetune")
    tokenizer.save_pretrained("./checkpoints/bert_squad_finetune")
    print("Training complete. Model saved to ./checkpoints/bert_squad_finetune")
