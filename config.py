import json
import torch
# with open('.data/wordmap.json', 'r') as j:
#     word_map = json.load(j)
# vocab_size = len(word_map)

load = False

vocab_size = 32000
max_len = 512
n_position = 512
batch_size = 32
model_dim = 768
ff_dim = 2048
head = 8
n_layers = 6
dropout_rate = 0.1
n_epochs = 3

seed = 116

learning_rate = 1e-5
betas = (0.9, 0.98)
max_grad_norm = 1.0

warmup = 128000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

bert_model_name = 'bert-base-uncased'

train_data = 'data/train_data.json'
train_data = 'data/train_data'
train_data_pickle_path = 'data/train_data.pkl'
data_dir = 'data'
fn = 'trained_model'
file_path = 'data/processed_cornell_data.txt'
