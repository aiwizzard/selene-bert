import torch
import numpy as np
from transformers.models.bert.tokenization_bert import BertTokenizer

import config as config
from model.model import ChatModel
from train_util import subsequent_mask


def evaluate(config, input_seq, tokenizer, model, device, verbose=True):
    model.eval()
    ids = tokenizer.encode(input_seq)
    src = torch.tensor(ids, dtype=torch.long, device=device).view(1, -1)
    src_mask = torch.ones(src.size(), dtype=torch.long, device=device)
    mem = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(tokenizer.cls_token_id).long().to(device)
    with torch.no_grad():
        for i in range(config.max_len - 1):
            out = model.decode(ys, mem, src_mask,
                             subsequent_mask(ys).type_as(ys))
            prob = model.generate(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.item()
            if next_word == tokenizer.sep_token_id:
                break
            ys = torch.cat([ys, torch.ones(1, 1).type_as(ys).fill_(next_word).long()], dim=1)
    ys = ys.view(-1).detach().cpu().numpy().tolist()[1:]
    text = tokenizer.decode(ys)
    if verbose:
        print(f'{text}')
    return text

if __name__ == '__main__':

    # device = torch.device(Config.device)
    device = torch.device('cpu')

    state_dict = torch.load(f'{config.data_dir}/{config.fn}.pth', map_location=device)

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)

    model = ChatModel(config).to(device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    # model.freeze()

    while True:
        s = input('You>')
        if s == 'q':
            break
        print('BOT>', end='')
        text = evaluate(config, s, tokenizer, model, device, True)