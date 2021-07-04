import torch
import numpy as np
import pickle
from tqdm import tqdm
import config as config

def create_train_data(tokenizer, use_pickle=False) -> list:
    data = []
    if use_pickle:
        with open(config.train_data_pickle_path, 'rb') as file:
            data = pickle.load(file)
    else:
        with open(config.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for i in tqdm(range(0, len(lines), 3)):
            li = []
            for line in lines[i: i+2]:
                li.append(line[:config.max_len])
            data.append(tuple(map(tokenizer.encode, li)))
        with open(config.train_data_pickle_path, 'wb') as file:
            pickle.dump(data, file)
    return data


def subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    _, size = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, size, size), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

# mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
# mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
# return mask


def create_masks(source, target, pad=0):
    source_mask = (source != pad).unsqueeze(-2).to(config.device)

    target_mask = (target != pad).unsqueeze(-2).to(config.device)
    target_mask = target_mask & subsequent_mask(target)

    return source_mask, target_mask

def seed_everything(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True