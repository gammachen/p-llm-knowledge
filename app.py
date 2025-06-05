from flask import Flask, request, jsonify
from predict import predict_relation, load_model, RelationPredictor
from data.loader import get_tokenizer
from config import CONFIG
import json

app = Flask(__name__)

# 全局变量
tokenizer = None
model = None
label_map = None

# 全局模型变量
loaded_model = None
model_loaded = False

tokenizer = get_tokenizer()

# 加载标签映射
try:
    with open(CONFIG["label_map_path"], 'r', encoding='utf-8', errors='ignore') as f:
        label_map = json.load(f)
    # 加载模型
    model = load_model(CONFIG['model_save_path'], len(label_map))
    model.eval()  # 设置为评估模式
except Exception as e:
    print(f"Error loading resources: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    ent1 = data.get('ent1')
    ent2 = data.get('ent2')
    
    if not all([text, ent1, ent2]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        relation, confidence = predict_relation(
            text, ent1, ent2, model, tokenizer, label_map
        )
        return jsonify({
            'text': text,
            'entity1': ent1,
            'entity2': ent2,
            'relation': relation,
            'confidence': float(confidence)
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)