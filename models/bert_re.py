import torch
import torch.nn as nn
from transformers import BertModel
from config import CONFIG

class BERTForRE(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(CONFIG['model_name'])
        # 调整词嵌入大小以适应新增的特殊标记
        self.bert.resize_token_embeddings(self.bert.config.vocab_size + len(CONFIG['special_tokens']))
        self.dropout = nn.Dropout(0.1) ## TODO 需要？
        self.classifier = nn.Linear(self.bert.config.hidden_size * 3, num_labels)

    def get_entity_repr(self, sequence_output, start, end):
        entity_repr = sequence_output[torch.arange(sequence_output.size(0)), start:end+1]
        return torch.mean(entity_repr, dim=1)

    def forward(self, input_ids, attention_mask, e1_start, e1_end, e2_start, e2_end):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 获取[CLS]标记表示
        cls_output = sequence_output[:, 0, :]
        
        # 实体位置增强
        e1_rep = sequence_output[torch.arange(sequence_output.size(0)), e1_start]
        e2_rep = sequence_output[torch.arange(sequence_output.size(0)), e2_start]
        
        # Concatenate all representations
        combined = torch.cat([cls_output, e1_rep, e2_rep], dim=1)
        
        logits = self.classifier(combined)
        return logits