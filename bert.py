import torch.nn as nn
from predictive_heads.head import get_pred_head
from transformers import (
    BertModel,
    BertPreTrainedModel
)

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.pred_head = get_pred_head(config)

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None, # index to the start of the "answer"
        end_positions=None # index to the end of the "answer"
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        seq_output = outputs.last_hidden_state # shape: (batch_size, seq_len, hidden_size)

        # get start and end logits from predictive head
        start_logits, end_logits = self.pred_head(seq_output) # shape: (batch_size, seq_len)

        total_loss = None # average of span start loss and span end loss

        if start_positions is not None and end_positions is not None:
            ignored_idx = start_logits.size(1)
            # (batch_size, seq_len) actual start/end labels (one 1 within the sequence for each item)
            start_positions = start_positions.clamp(0, ignored_idx) 
            end_positions = end_positions.clamp(0, ignored_idx)

            loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_idx)
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2.0

        return {
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits
        }