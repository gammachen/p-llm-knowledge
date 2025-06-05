import torch
from torch.utils.data import Dataset
from utils.preprocess import preprocess, tokenize_with_entities

class REDataset(Dataset):
    def __init__(self, texts, ent1s, ent2s, relations, tokenizer, label_map):
        self.texts = texts
        self.ent1s = ent1s
        self.ent2s = ent2s
        self.relations = relations
        self.tokenizer = tokenizer
        self.label_map = label_map

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = preprocess(self.texts[idx], self.ent1s[idx], self.ent2s[idx])
        inputs = tokenize_with_entities(text, self.tokenizer)
        label = torch.tensor(self.label_map[self.relations[idx]])
        return {**inputs, 'labels': label}