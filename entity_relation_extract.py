import re
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(text, ent1, ent2):
    """插入实体位置标记"""
    text = text.replace(ent1, f"[E1]{ent1}[/E1]", 1)
    text = text.replace(ent2, f"[E2]{ent2}[/E2]", 1)
    return text

# 示例数据加载 (需替换为真实数据)
data = relation_data = [
    # 公司关系 (10条)
    {"text": "Apple was founded by Steve Jobs in Cupertino", "ent1": "Apple", "ent2": "Steve Jobs", "relation": "founded_by"},
    {"text": "Microsoft is headquartered in Redmond, Washington", "ent1": "Microsoft", "ent2": "Redmond", "relation": "headquartered_in"},
    {"text": "Elon Musk serves as the CEO of Tesla", "ent1": "Elon Musk", "ent2": "Tesla", "relation": "ceo_of"},
    {"text": "Amazon acquired Whole Foods in 2017", "ent1": "Amazon", "ent2": "Whole Foods", "relation": "acquired"},
    {"text": "Google's parent company is Alphabet Inc.", "ent1": "Google", "ent2": "Alphabet", "relation": "subsidiary_of"},
    {"text": "Netflix competes directly with Disney+", "ent1": "Netflix", "ent2": "Disney+", "relation": "competitor_of"},
    {"text": "IBM manufactures mainframe computers", "ent1": "IBM", "ent2": "mainframe computers", "relation": "manufactures"},
    {"text": "Samsung is based in Seoul, South Korea", "ent1": "Samsung", "ent2": "Seoul", "relation": "located_in"},
    {"text": "Tim Cook succeeded Steve Jobs at Apple", "ent1": "Tim Cook", "ent2": "Steve Jobs", "relation": "succeeded"},
    {"text": "Intel supplies processors to Dell", "ent1": "Intel", "ent2": "Dell", "relation": "supplier_of"},
    
    # 地理关系 (15条)
    {"text": "Paris is the capital city of France", "ent1": "Paris", "ent2": "France", "relation": "capital_of"},
    {"text": "The Nile River flows through Egypt", "ent1": "Nile River", "ent2": "Egypt", "relation": "flows_through"},
    {"text": "Mount Everest is located in the Himalayas", "ent1": "Mount Everest", "ent2": "Himalayas", "relation": "part_of"},
    {"text": "Tokyo is situated on Honshu Island", "ent1": "Tokyo", "ent2": "Honshu Island", "relation": "located_on"},
    {"text": "The Sahara Desert spans across North Africa", "ent1": "Sahara Desert", "ent2": "North Africa", "relation": "located_in"},
    {"text": "Venice is famous for its canals", "ent1": "Venice", "ent2": "canals", "relation": "known_for"},
    {"text": "The Great Barrier Reef is adjacent to Australia", "ent1": "Great Barrier Reef", "ent2": "Australia", "relation": "adjacent_to"},
    {"text": "The Amazon River originates in the Andes", "ent1": "Amazon River", "ent2": "Andes", "relation": "originates_in"},
    {"text": "New York City contains five boroughs", "ent1": "New York City", "ent2": "boroughs", "relation": "contains"},
    {"text": "The Grand Canyon was formed by the Colorado River", "ent1": "Grand Canyon", "ent2": "Colorado River", "relation": "formed_by"},
    {"text": "Singapore is an island country", "ent1": "Singapore", "ent2": "island", "relation": "is"},
    {"text": "The Pacific Ocean borders California", "ent1": "Pacific Ocean", "ent2": "California", "relation": "borders"},
    {"text": "Mount Fuji is the highest mountain in Japan", "ent1": "Mount Fuji", "ent2": "Japan", "relation": "located_in"},
    {"text": "The Danube River empties into the Black Sea", "ent1": "Danube River", "ent2": "Black Sea", "relation": "flows_into"},
    {"text": "The Eiffel Tower stands in Paris", "ent1": "Eiffel Tower", "ent2": "Paris", "relation": "located_in"},
    
    # 人物关系 (20条)
    {"text": "Albert Einstein developed the theory of relativity", "ent1": "Albert Einstein", "ent2": "theory of relativity", "relation": "developed"},
    {"text": "Marie Curie won the Nobel Prize in Physics", "ent1": "Marie Curie", "ent2": "Nobel Prize", "relation": "awarded"},
    {"text": "William Shakespeare wrote Hamlet", "ent1": "William Shakespeare", "ent2": "Hamlet", "relation": "author_of"},
    {"text": "Leonardo da Vinci painted the Mona Lisa", "ent1": "Leonardo da Vinci", "ent2": "Mona Lisa", "relation": "creator_of"},
    {"text": "Martin Luther King Jr. led the Civil Rights Movement", "ent1": "Martin Luther King Jr.", "ent2": "Civil Rights Movement", "relation": "leader_of"},
    {"text": "Steve Jobs co-founded Apple with Steve Wozniak", "ent1": "Steve Jobs", "ent2": "Steve Wozniak", "relation": "co-founder_with"},
    {"text": "Elon Musk was born in South Africa", "ent1": "Elon Musk", "ent2": "South Africa", "relation": "born_in"},
    {"text": "Mahatma Gandhi advocated nonviolent resistance", "ent1": "Mahatma Gandhi", "ent2": "nonviolent resistance", "relation": "advocated"},
    {"text": "Nelson Mandela was imprisoned on Robben Island", "ent1": "Nelson Mandela", "ent2": "Robben Island", "relation": "imprisoned_at"},
    {"text": "Michelangelo sculpted David", "ent1": "Michelangelo", "ent2": "David", "relation": "sculpted"},
    {"text": "Charles Darwin proposed the theory of evolution", "ent1": "Charles Darwin", "ent2": "theory of evolution", "relation": "proposed"},
    {"text": "Stephen Hawking studied black holes", "ent1": "Stephen Hawking", "ent2": "black holes", "relation": "studied"},
    {"text": "Amelia Earhart disappeared over the Pacific Ocean", "ent1": "Amelia Earhart", "ent2": "Pacific Ocean", "relation": "disappeared_over"},
    {"text": "Thomas Edison invented the phonograph", "ent1": "Thomas Edison", "ent2": "phonograph", "relation": "invented"},
    {"text": "Pablo Picasso founded the Cubist movement", "ent1": "Pablo Picasso", "ent2": "Cubist movement", "relation": "founded"},
    {"text": "Coco Chanel revolutionized women's fashion", "ent1": "Coco Chanel", "ent2": "women's fashion", "relation": "revolutionized"},
    {"text": "Isaac Newton formulated the laws of motion", "ent1": "Isaac Newton", "ent2": "laws of motion", "relation": "formulated"},
    {"text": "Winston Churchill was the Prime Minister of the UK", "ent1": "Winston Churchill", "ent2": "UK", "relation": "leader_of"},
    {"text": "Cleopatra ruled ancient Egypt", "ent1": "Cleopatra", "ent2": "ancient Egypt", "relation": "ruled"},
    {"text": "Neil Armstrong walked on the Moon", "ent1": "Neil Armstrong", "ent2": "Moon", "relation": "walked_on"},
    
    # 科技与科学 (15条)
    {"text": "DNA contains genetic information", "ent1": "DNA", "ent2": "genetic information", "relation": "contains"},
    {"text": "Photosynthesis occurs in plant chloroplasts", "ent1": "Photosynthesis", "ent2": "chloroplasts", "relation": "occurs_in"},
    {"text": "The Internet relies on TCP/IP protocols", "ent1": "Internet", "ent2": "TCP/IP", "relation": "depends_on"},
    {"text": "CRISPR is used for gene editing", "ent1": "CRISPR", "ent2": "gene editing", "relation": "used_for"},
    {"text": "Blockchain technology underpins cryptocurrencies", "ent1": "Blockchain", "ent2": "cryptocurrencies", "relation": "underpins"},
    {"text": "Artificial intelligence includes machine learning", "ent1": "Artificial intelligence", "ent2": "machine learning", "relation": "includes"},
    {"text": "Neural networks mimic the human brain", "ent1": "Neural networks", "ent2": "human brain", "relation": "mimics"},
    {"text": "The Large Hadron Collider discovered the Higgs boson", "ent1": "Large Hadron Collider", "ent2": "Higgs boson", "relation": "discovered"},
    {"text": "Quantum computing uses qubits", "ent1": "Quantum computing", "ent2": "qubits", "relation": "uses"},
    {"text": "Vaccines stimulate the immune system", "ent1": "Vaccines", "ent2": "immune system", "relation": "stimulates"},
    {"text": "Global warming causes climate change", "ent1": "Global warming", "ent2": "climate change", "relation": "causes"},
    {"text": "Renewable energy includes solar power", "ent1": "Renewable energy", "ent2": "solar power", "relation": "includes"},
    {"text": "Black holes have event horizons", "ent1": "Black holes", "ent2": "event horizons", "relation": "has"},
    {"text": "Autonomous vehicles use lidar sensors", "ent1": "Autonomous vehicles", "ent2": "lidar sensors", "relation": "uses"},
    {"text": "mRNA technology enabled COVID-19 vaccines", "ent1": "mRNA technology", "ent2": "COVID-19 vaccines", "relation": "enabled"},
    
    # 历史事件 (10条)
    {"text": "World War II ended in 1945", "ent1": "World War II", "ent2": "1945", "relation": "ended_in"},
    {"text": "The Industrial Revolution began in Britain", "ent1": "Industrial Revolution", "ent2": "Britain", "relation": "began_in"},
    {"text": "The Renaissance originated in Italy", "ent1": "Renaissance", "ent2": "Italy", "relation": "originated_in"},
    {"text": "The French Revolution overthrew the monarchy", "ent1": "French Revolution", "ent2": "monarchy", "relation": "overthrew"},
    {"text": "The Apollo program landed astronauts on the Moon", "ent1": "Apollo program", "ent2": "Moon", "relation": "landed_on"},
    {"text": "The Berlin Wall separated East and West Berlin", "ent1": "Berlin Wall", "ent2": "Berlin", "relation": "separated"},
    {"text": "The Titanic sank in the Atlantic Ocean", "ent1": "Titanic", "ent2": "Atlantic Ocean", "relation": "sank_in"},
    {"text": "The Magna Carta limited the king's power", "ent1": "Magna Carta", "ent2": "king", "relation": "limited"},
    {"text": "The Cold War involved the US and USSR", "ent1": "Cold War", "ent2": "US", "relation": "involved"},
    {"text": "The Declaration of Independence was signed in 1776", "ent1": "Declaration of Independence", "ent2": "1776", "relation": "signed_in"},
    
    # 文化艺术 (15条)
    {"text": "The Louvre Museum houses the Mona Lisa", "ent1": "Louvre Museum", "ent2": "Mona Lisa", "relation": "houses"},
    {"text": "J.K. Rowling authored the Harry Potter series", "ent1": "J.K. Rowling", "ent2": "Harry Potter", "relation": "author_of"},
    {"text": "The Beatles originated from Liverpool", "ent1": "The Beatles", "ent2": "Liverpool", "relation": "originated_from"},
    {"text": "Star Wars was created by George Lucas", "ent1": "Star Wars", "ent2": "George Lucas", "relation": "created_by"},
    {"text": "Bollywood is based in Mumbai", "ent1": "Bollywood", "ent2": "Mumbai", "relation": "based_in"},
    {"text": "The Sistine Chapel features Michelangelo's frescoes", "ent1": "Sistine Chapel", "ent2": "Michelangelo", "relation": "features"},
    {"text": "Shakespeare's Globe Theatre is in London", "ent1": "Globe Theatre", "ent2": "London", "relation": "located_in"},
    {"text": "The Olympic Games include summer and winter events", "ent1": "Olympic Games", "ent2": "summer events", "relation": "includes"},
    {"text": "Hip hop music emerged from New York City", "ent1": "Hip hop", "ent2": "New York City", "relation": "emerged_from"},
    {"text": "The Great Gatsby was written by F. Scott Fitzgerald", "ent1": "The Great Gatsby", "ent2": "F. Scott Fitzgerald", "relation": "written_by"},
    {"text": "Mona Lisa depicts Lisa Gherardini", "ent1": "Mona Lisa", "ent2": "Lisa Gherardini", "relation": "depicts"},
    {"text": "The Statue of Liberty was a gift from France", "ent1": "Statue of Liberty", "ent2": "France", "relation": "gift_from"},
    {"text": "The Harry Potter series involves magic", "ent1": "Harry Potter", "ent2": "magic", "relation": "involves"},
    {"text": "The Sydney Opera House hosts performances", "ent1": "Sydney Opera House", "ent2": "performances", "relation": "hosts"},
    {"text": "Impressionism was pioneered by Claude Monet", "ent1": "Impressionism", "ent2": "Claude Monet", "relation": "pioneered_by"},
    
    # 医学健康 (15条)
    {"text": "Insulin regulates blood sugar levels", "ent1": "Insulin", "ent2": "blood sugar", "relation": "regulates"},
    {"text": "COVID-19 is caused by SARS-CoV-2", "ent1": "COVID-19", "ent2": "SARS-CoV-2", "relation": "caused_by"},
    {"text": "Antibiotics treat bacterial infections", "ent1": "Antibiotics", "ent2": "bacterial infections", "relation": "treats"},
    {"text": "The heart pumps blood", "ent1": "heart", "ent2": "blood", "relation": "pumps"},
    {"text": "MRI scans produce detailed images", "ent1": "MRI", "ent2": "images", "relation": "produces"},
    {"text": "Vaccines prevent infectious diseases", "ent1": "Vaccines", "ent2": "infectious diseases", "relation": "prevents"},
    {"text": "Chemotherapy targets cancer cells", "ent1": "Chemotherapy", "ent2": "cancer cells", "relation": "targets"},
    {"text": "DNA stores genetic information", "ent1": "DNA", "ent2": "genetic information", "relation": "stores"},
    {"text": "The lungs exchange oxygen and carbon dioxide", "ent1": "lungs", "ent2": "oxygen", "relation": "exchanges"},
    {"text": "Penicillin was discovered by Alexander Fleming", "ent1": "Penicillin", "ent2": "Alexander Fleming", "relation": "discovered_by"},
    {"text": "The brain controls the nervous system", "ent1": "brain", "ent2": "nervous system", "relation": "controls"},
    {"text": "Antibodies fight infections", "ent1": "Antibodies", "ent2": "infections", "relation": "fights"},
    {"text": "Stem cells differentiate into specialized cells", "ent1": "Stem cells", "ent2": "specialized cells", "relation": "differentiate_into"},
    {"text": "The pancreas produces insulin", "ent1": "pancreas", "ent2": "insulin", "relation": "produces"},
    {"text": "Radiotherapy treats tumors", "ent1": "Radiotherapy", "ent2": "tumors", "relation": "treats"}
]

df = pd.DataFrame(data)
df["processed_text"] = df.apply(lambda x: preprocess(
    x["text"], x["ent1"], x["ent2"]), axis=1)

# 划分数据集
train_df, test_df = train_test_split(df, test_size=0.2)

# 初始化 tokenizer
from transformers import AutoTokenizer

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

# 添加特殊标记
special_tokens = {"additional_special_tokens": ["[E1]", "[E2]", "[/E1]", "[/E2]"]}
tokenizer.add_special_tokens(special_tokens)

# 确认 token 是否存在
assert tokenizer.convert_tokens_to_ids("[E1]") != 100, "[E1] token 被错误映射为 100"

def tokenize_with_entities(text):
    # 分词时保留特殊标记
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
    return encoding
    
from torch.utils.data import Dataset, DataLoader
import torch

class REDataset(Dataset):
    def __init__(self, dataframe, label_map):
        self.data = dataframe
        self.label_map = label_map
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        tokenized = tokenize_with_entities(sample["processed_text"])
        label = self.label_map.get(sample["relation"], -1)
        return {
            **tokenized,
            "labels": torch.tensor(label)
        }

# 创建标签映射
relations = df["relation"].unique()
label_map = {rel: idx for idx, rel in enumerate(relations)}

# 创建DataLoader
train_dataset = REDataset(train_df, label_map)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

from transformers import BertModel
import torch.nn as nn

class BERTForRE(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.resize_token_embeddings(len(tokenizer))  # 适配新token
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, e1_start, e1_end, e2_start, e2_end):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 获取[CLS]标记表示
        cls_output = sequence_output[:, 0, :]
        
        # 实体位置增强（可选）
        e1_rep = sequence_output[torch.arange(sequence_output.size(0)), e1_start]
        e2_rep = sequence_output[torch.arange(sequence_output.size(0)), e2_start]
        combined = torch.cat([cls_output, e1_rep, e2_rep], dim=1)
        
        logits = self.classifier(combined)
        return logits

model = BERTForRE(num_labels=len(label_map))

from transformers.optimization import Adafactor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = Adafactor(
    model.parameters(), 
    lr=1e-5, 
    scale_parameter=False, 
    relative_step=False)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(5):
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

def custom_collate_fn(batch):
    # 调试输出：打印每个样本的 input_ids 和 attention_mask 形状
    print("Batch input_ids shapes:", [item['input_ids'].shape for item in batch])
    print("Batch attention_mask shapes:", [item['attention_mask'].shape for item in batch])
    
    # 提取 input_ids 并进行填充
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    
    # 提取 attention_mask 并进行填充
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    
    # 提取 labels
    labels = torch.tensor([item['labels'] for item in batch])
    
    # 调试输出：填充后的张量形状
    print("Padded input_ids shape:", input_ids.shape)
    print("Padded attention_mask shape:", attention_mask.shape)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

# 确保 Dataset 返回的每个样本结构正确
class YourDatasetClass(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokenized = tokenize_with_entities(sample["processed_text"])
        
        # 确保 input_ids 和 attention_mask 的形状为 (seq_length,)
        return {
            'input_ids': tokenized["input_ids"].squeeze(0),  # 去掉 batch 维度
            'attention_mask': tokenized["attention_mask"].squeeze(0),  # 去掉 batch 维度
            'labels': sample["label"]
        }

def predict_relation(text, ent1, ent2):
    processed = preprocess(text, ent1, ent2)
    inputs = tokenize_with_entities(processed)
    inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        logits = model(**inputs)
    
    probs = torch.softmax(logits, dim=-1)
    pred_idx = torch.argmax(probs).item()
    
    # 反转标签映射
    idx_to_label = {v: k for k, v in label_map.items()}
    return idx_to_label[pred_idx], probs[0][pred_idx].item()

# 测试
text = "Tim Cook is the CEO of Apple"
ent1, ent2 = "Tim Cook", "Apple"
relation, confidence = predict_relation(text, ent1, ent2)
print(f"Relation: {relation} (Confidence: {confidence:.2f})")

