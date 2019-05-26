import torch
import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel


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
                attention_mask: torch.Tensor,  # (b, seq)
                ) -> torch.Tensor:  # (b, label)
        # (b, hid)
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, output_all_encoded_layers=False)
        logits = self.classifier(self.dropouot(pooled_output))
        return logits
