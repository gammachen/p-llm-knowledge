import torch
from config import CONFIG

def preprocess(text, ent1, ent2):
    """插入实体位置标记"""
    text = text.replace(ent1, f"[E1]{ent1}[/E1]", 1)
    text = text.replace(ent2, f"[E2]{ent2}[/E2]", 1)
    return text

def tokenize_with_entities(text, tokenizer):
    # Tokenize the text and get input IDs
    encoding = tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)
    
    # Get special token IDs
    e1_id = tokenizer.convert_tokens_to_ids("[E1]")
    e2_id = tokenizer.convert_tokens_to_ids("[E2]")
    e1_end_id = tokenizer.convert_tokens_to_ids("[/E1]")
    e2_end_id = tokenizer.convert_tokens_to_ids("[/E2]")
    
    # Find positions of entity markers
    e1_start = (input_ids == e1_id).nonzero(as_tuple=True)[0].item() if e1_id in input_ids else -1
    e2_start = (input_ids == e2_id).nonzero(as_tuple=True)[0].item() if e2_id in input_ids else -1
    e1_end = (input_ids == e1_end_id).nonzero(as_tuple=True)[0].item() if e1_end_id in input_ids else -1
    e2_end = (input_ids == e2_end_id).nonzero(as_tuple=True)[0].item() if e2_end_id in input_ids else -1
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "e1_start": e1_start,
        "e2_start": e2_start,
        "e1_end": e1_end,
        "e2_end": e2_end
    }