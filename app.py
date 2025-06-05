from flask import Flask, request, jsonify, render_template
from predict import predict_relation, load_model, RelationPredictor
from data.loader import get_tokenizer
from config import CONFIG
import json

app = Flask(__name__)

# 初始化模型和相关资源
def init_resources():
    """初始化模型、分词器和标签映射"""
    try:
        # 获取分词器
        tokenizer = get_tokenizer()
        
        # 加载标签映射
        with open(CONFIG["label_map_path"], 'r', encoding='utf-8', errors='ignore') as f:
            label_map = json.load(f)
        
        # 加载模型
        model = load_model(CONFIG['model_save_path'], len(label_map))
        model.eval()  # 设置为评估模式
        
        return tokenizer, model, label_map
        
    except FileNotFoundError as e:
        print(f"File not found error during initialization: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        print(f"JSON decode error in label map: {str(e)}")
        raise
    except Exception as e:
        print(f"Unexpected error during initialization: {str(e)}")
        raise

# 初始化全局资源
try:
    tokenizer, model, label_map = init_resources()
except Exception as e:
    print(f"Failed to initialize resources: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    try:
        # 获取请求数据
        data = request.json
        text = data.get('text')
        ent1 = data.get('ent1')
        ent2 = data.get('ent2')
        
        # 验证必需参数
        if not all([text, ent1, ent2]):
            return jsonify({
                'text': text or '',
                'entity1': ent1 or '',
                'entity2': ent2 or '',
                'relation': '',
                'confidence': 0.0,
                'error': 'Missing required parameters'
            }), 400
            
        # 执行预测
        relation, confidence = predict_relation(
            text, ent1, ent2, model, tokenizer, label_map
        )
        
        # 返回成功响应
        return jsonify({
            'text': text,
            'entity1': ent1,
            'entity2': ent2,
            'relation': relation,
            'confidence': float(confidence),
            'error': ''
        })
        
    except Exception as e:
        # 记录错误详情
        print(f"Error during prediction: {str(e)}")
        
        # 返回带有详细错误信息的响应
        return jsonify({
            'text': text or '',
            'entity1': ent1 or '',
            'entity2': ent2 or '',
            'relation': '',
            'confidence': 0.0,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)