import torch
import pandas as pd
from flask import Flask, request, render_template_string
from entity_relation_extract import predict_relation, load_model, label_map

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
num_labels = len(label_map)
model = load_model(num_labels)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    ent1 = request.form['ent1']
    ent2 = request.form['ent2']
    
    relation, confidence = predict_relation(text, ent1, ent2)
    return f"关系: {relation} (置信度: {confidence:.2f})"

@app.route('/')
def index():
    return '''
    <h1>关系抽取系统</h1>
    <form action="/predict" method="POST">
        <label>输入句子:</label><br>
        <input type="text" name="text" size="80"><br>
        <label>实体1:</label>
        <input type="text" name="ent1">
        <label>实体2:</label>
        <input type="text" name="ent2"><br>
        <input type="submit" value="抽取关系">
    </form>
    '''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)