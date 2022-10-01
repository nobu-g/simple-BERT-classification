import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class BertClassifier(nn.Module):
    def __init__(self,
                 bert_model: str,
                 num_labels: int
                 ) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                ) -> torch.Tensor:  # (b, label)
        output: BaseModelOutputWithPoolingAndCrossAttentions
        output = self.bert(input_ids,
                           token_type_ids=segment_ids,
                           attention_mask=attention_mask,
                           output_all_encoded_layers=False)
        logits = self.classifier(self.dropout(output.pooler_output))
        return logits
