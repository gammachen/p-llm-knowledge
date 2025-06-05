import torch
from utils.preprocess import preprocess, tokenize_with_entities
from data.loader import get_tokenizer
from models.bert_re import BERTForRE
import json

class RelationPredictor:
    def __init__(self, model, label_map):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = get_tokenizer()
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self.label_map = label_map
        self.idx_to_label = {v: k for k, v in label_map.items()}

    def predict(self, text, ent1, ent2):
        processed = preprocess(text, ent1, ent2)
        inputs = tokenize_with_entities(processed, self.tokenizer)
        
        # Convert integer values to tensors before unsqueezing
        tensor_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, int):
                tensor_inputs[k] = torch.tensor([v]).to(self.device)
            else:
                tensor_inputs[k] = v.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**tensor_inputs)
        
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        
        return self.idx_to_label[pred_idx], probs[0][pred_idx].item()

def load_model(model_path, num_labels):
    """加载预训练模型"""
    model = BERTForRE(num_labels)
    model.load_state_dict(torch.load(model_path))
    return model

def predict_relation(text, ent1, ent2, model, tokenizer, label_map):
    """预测关系的包装函数"""
    predictor = RelationPredictor(model, label_map)
    predictor.tokenizer = tokenizer  # 使用传入的tokenizer
    
    print(f" Predicting relation for text: {text}, ent1: {ent1}, ent2: {ent2}")
    return predictor.predict(text, ent1, ent2)

def test_predictor():
    # 加载实际的模型和标签映射
    label_map = json.load(open("saved_models/label_map.json"))
    model = load_model("saved_models/pytorch_model.bin", len(label_map))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    predictor = RelationPredictor(model, label_map)
    
    # 测试英文样例
    text = "Tim Cook is the CEO of Apple"
    ent1, ent2 = "Tim Cook", "Apple"
    relation, confidence = predictor.predict(text, ent1, ent2)
    print(f"Relation: {relation} (Confidence: {confidence:.2f})")
    
    # 测试中文样例
    text = "云南白药具有止血化瘀特别的作用"
    ent1, ent2 = "云南白药", "止血化瘀"
    relation, confidence = predictor.predict(text, ent1, ent2)
    print(f"Relation: {relation} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    test_predictor()