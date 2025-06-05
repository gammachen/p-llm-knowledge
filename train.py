import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.optimization import Adafactor
import os
from data.loader import load_data, create_label_map, get_tokenizer
from data.dataset import REDataset
from models.bert_re import BERTForRE
from config import CONFIG
import json

def load_model(num_labels, model_path="saved_models"):
    """使用PyTorch原生方法加载模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTForRE(num_labels=num_labels)
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()
    return model

def custom_collate_fn(batch, tokenizer):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    labels = torch.tensor([item['labels'] for item in batch])
    e1_start = torch.tensor([item['e1_start'] for item in batch])
    e2_start = torch.tensor([item['e2_start'] for item in batch])
    e1_end = torch.tensor([item['e1_end'] for item in batch])
    e2_end = torch.tensor([item['e2_end'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'e1_start': e1_start,
        'e2_start': e2_start,
        'e1_end': e1_end,
        'e2_end': e2_end
    }

def save_model(model, save_dir="saved_models"):
    """使用PyTorch原生方法保存模型"""
    os.makedirs(save_dir, exist_ok=True)
    # 保存模型状态字典
    torch.save(model.state_dict(), CONFIG['model_save_path'])
    # 保存BERT配置文件
    model.bert.config.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

def save_label_map(label_map, save_dir="saved_models"):
    """保存label_map为JSON文件"""
    os.makedirs(save_dir, exist_ok=True)
    with open(CONFIG['label_map_path'], 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"Label map saved to {CONFIG['label_map_path']}")

def train(data_path, num_epochs=5, batch_size=8, learning_rate=1e-5):
    # 加载数据
    texts, ent1s, ent2s, relations = load_data(data_path)
    
    # 创建标签映射
    label_map = create_label_map(relations)
    
    # 获取tokenizer
    tokenizer = get_tokenizer()
    
    # 创建数据集
    dataset = REDataset(texts, ent1s, ent2s, relations, tokenizer, label_map)
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: custom_collate_fn(x, tokenizer)
    )

    # 初始化模型
    model = BERTForRE(num_labels=len(label_map))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 优化器和损失函数
    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        scale_parameter=False,
        relative_step=False
    )
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                e1_start=batch["e1_start"],
                e1_end=batch["e1_end"],
                e2_start=batch["e2_start"],
                e2_end=batch["e2_end"]
            )
            
            loss = criterion(outputs, batch["labels"])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

    # 保存模型
    save_model(model)
    
    # 保存label_map
    save_label_map(label_map)
    
    return model, label_map

if __name__ == "__main__":
    train("data.json")
