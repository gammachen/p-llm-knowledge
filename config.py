import torch

CONFIG = {
    'model_name': 'bert-base-chinese',
    'max_length': 128,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_save_path': 'saved_models/pytorch_model.bin',
    'label_map_path':'saved_models/label_map.json',
    'special_tokens': ['[E1]', '[/E1]', '[E2]', '[/E2]']
}