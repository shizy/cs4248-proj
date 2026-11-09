import json
from torch.utils.data import Dataset
import torch
import random

# Tokenization + Alignment
def parse_squad_json(filepath):
    """Flattens the nested SQuAD JSON structure."""
    with open(filepath, "r", encoding="utf-8") as f:
        squad_dict = json.load(f)

    data_rows = []
    for article in squad_dict["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                q_id = qa["id"]
                question = qa["question"]
                answers = qa.get("answers", [])
                
                data_rows.append({
                    "id": q_id,
                    "question": question,
                    "context": context,
                    "answers": answers,
                })
    return data_rows


class SquadDataset(Dataset):
    def __init__(
        self,
        data_rows,
        tokenizer,
        max_length=384,
        doc_stride=128,
        is_train=True,
        max_train_rows=None,
        sample_seed=42
    ):
        self.is_train = is_train
        self.features = []

        if is_train and max_train_rows is not None:
            if max_train_rows <= 0:
                raise ValueError("max_train_rows must be positive")

            if len(data_rows) > max_train_rows:
                data_rows = random.Random(sample_seed).sample(
                    data_rows, max_train_rows
                )

        print(f"Tokenizing {len(data_rows):,} examples", flush=True)
        print(f"tokenizer={tokenizer.name_or_path}, "
            f"max_length={max_length}, doc_stride={doc_stride}",
            flush=True,
        )

        questions = [" ".join(row["question"].split()) for row in data_rows]
        contexts = [row["context"] for row in data_rows]
        
        try:
            tokenized = tokenizer(
                questions,
                contexts,
                truncation="only_second",
                max_length=max_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_token_type_ids=True, # RoBERTa's tokenizer omits these by default; they are all zeros there
                padding="max_length",
                return_tensors="pt",
            )
        except:
            for row in data_rows:
                try:
                    tokenizer(
                        row["question"],
                        row["context"],
                        truncation="only_second",
                        max_length=max_length,
                        stride=doc_stride,
                        return_overflowing_tokens=True,
                        return_offsets_mapping=True,
                        return_token_type_ids=True,
                        padding="max_length",
                        return_tensors="pt"
                    )
                # Debugging Stuff... 
                except Exception as exc:
                    question_length = len(tokenizer(
                        row["question"],
                        add_special_tokens=False,
                        truncation=False,
                    )["input_ids"])

                    special_tokens = tokenizer.num_special_tokens_to_add(
                        pair=True
                    )

                    raise RuntimeError(
                        f"Failed example: {row['id']}\n"
                        f"Question tokens: {question_length}\n"
                        f"Context budget: "
                        f"{max_length - question_length - special_tokens}\n"
                        f"max_length={max_length}, stride={doc_stride}\n"
                        f"Question: {row['question']!r}"
                    ) from exc
            raise

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        for i, offsets in enumerate(offset_mapping):
            sample_index = sample_mapping[i].item()
            row = data_rows[sample_index]
            input_ids = tokenized["input_ids"][i]
            sequence_ids = tokenized.sequence_ids(i)

            item = {
                "input_ids": input_ids,
                "attention_mask": tokenized["attention_mask"][i],
                "token_type_ids": tokenized["token_type_ids"][i],
            }

            if self.is_train:
                # Calculate target positions for loss calculation
                cls_index = (input_ids == tokenizer.cls_token_id).nonzero(
                    as_tuple=True
                )[0].item() # squad 2.0 addition: unanswerable questions are defaulted to the cls token
                answers = row["answers"]

                if not answers:
                    start_pos, end_pos = cls_index, cls_index # squad 2.0 addition for no answers: set span start and end to the cls token index
                else:
                    start_char = answers[0]["answer_start"]
                    end_char = start_char + len(answers[0]["text"])

                    token_start_index = 0
                    while sequence_ids[token_start_index] != 1:
                        token_start_index += 1

                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != 1:
                        token_end_index -= 1

                    if not (
                        offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char
                    ):
                        start_pos, end_pos = cls_index, cls_index # squad 2.0 addition: answer outside of window, set span start and end to cls token index
                    else:
                        curr_idx = token_start_index
                        while (
                            curr_idx <= token_end_index and offsets[curr_idx][0] <= start_char
                        ):
                            curr_idx += 1
                        start_pos = curr_idx - 1

                        curr_idx = token_end_index
                        while (
                            curr_idx >= token_start_index and offsets[curr_idx][1] >= end_char
                        ):
                            curr_idx -= 1
                        end_pos = curr_idx + 1

                item["start_positions"] = torch.tensor(start_pos, dtype=torch.long)
                item["end_positions"] = torch.tensor(end_pos, dtype=torch.long)
            else:
                # Store metadata required for prediction & post-processing
                item["offset_mapping"] = offsets.tolist()
                item["sequence_ids"] = sequence_ids
                item["q_id"] = row["id"]
                item["context"] = row["context"]

            self.features.append(item)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
