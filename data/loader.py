import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from config import CONFIG
from transformers import AutoTokenizer

def load_data(data_path):
    # 这里假设数据是JSON格式，可以根据实际情况修改
    df = pd.read_json(data_path)
    return df['text'].tolist(), df['ent1'].tolist(), \
           df['ent2'].tolist(), df['relation'].tolist()

def create_label_map(relations):
    unique_relations = list(set(relations))
    return {rel: idx for idx, rel in enumerate(unique_relations)}

def get_tokenizer():
    # tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.add_special_tokens({'additional_special_tokens': CONFIG['special_tokens']})
    return tokenizer