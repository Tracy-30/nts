from torch.utils.data import DataLoader
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

import torch
import numpy as np
from config import cfg
from dataset.cnn_dm import CNNDM, CNNDM_SMALL
from dataset.tldr_news import TLDR_NEWS

def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = '{}/{}'.format(cfg['data_path'],data_name)

    if data_name in ['CNN_DAILYMAILS']:
        tokenizer = 'bert-base-uncased'
        dataset['train'] = eval('{}(root=root, split=\'train\', tokenizer=tokenizer)'.format('CNNDM'))
        dataset['test'] = eval('{}(root=root, split=\'test\', tokenizer=tokenizer)'.format('CNNDM'))

    elif data_name in ['CNN_DAILYMAILS_SMALL']:
        tokenizer = 'bert-base-uncased'
        dataset['train'] = eval('{}(root=root, split=\'train\', tokenizer=tokenizer)'.format('CNNDM_SMALL'))
        dataset['test'] = eval('{}(root=root, split=\'test\', tokenizer=tokenizer)'.format('CNNDM_SMALL'))

    elif data_name in ['TLDR_NEWS']:
        tokenizer = 'bert-base-uncased'
        dataset['train'] = eval('{}(root=root, split=\'train\', tokenizer=tokenizer)'.format('TLDR_NEWS'))
        dataset['test'] = eval('{}(root=root, split=\'test\', tokenizer=tokenizer)'.format('TLDR_NEWS'))

    elif data_name in ['BBC_NEWS']:
        tokenizer = 'bert-base-uncased'
        dataset['train'] = eval('{}(root=root, split=\'train\', tokenizer=tokenizer)'.format('BBC_NEWS'))
        dataset['test'] = eval('{}(root=root, split=\'test\', tokenizer=tokenizer)'.format('BBC_NEWS'))

    else:
        raise ValueError('Not valid dataset name')

    print('data ready')
    return dataset

def input_collate(batch):
    output = {key: [] for key in batch[0].keys()}
    for b in batch:
        for key in b:
            output[key].append(b[key])
    for key in output:
        if key != 'target_text':
            output[key] = torch.stack(output[key])
    return output 

def make_data_loader(dataset, tag, sampler=None):
    data_loader = {}
    for split in dataset:
        _batch_size = cfg[tag]['batch_size'][split] 
        _shuffle = cfg[tag]['shuffle'][split]
        
        if sampler is None:
            data_loader[split] = DataLoader(dataset=dataset[split], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[split] = DataLoader(dataset=dataset[split], batch_size=_batch_size, sampler=sampler[split],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


if __name__ == "__main__":
    dataset = fetch_dataset(cfg['data_name'])
    data_loader = make_data_loader(dataset,'two_stage_summarizer')

    print(len(data_loader['train']), len(data_loader['test']))
    

    for batch, input in enumerate(data_loader['train']):
        print(input.keys())

        print(len(input['input_ids']))

        print(input['input_ids'][0])
        print(input['input_mask'][0])
        print(input['input_type_ids'][0])

        print(input['target_ids'][0])
        print(input['target_mask'][0])
        print(input['target_type_ids'][0])

        print(input['target_text'])


        # print("data_piece_1:     ")
        # print(input['input_ids'][0])
        # print(tokenizer.convert_ids_to_tokens(input['input_ids'][0].tolist()))
        # # print(input['ipt_mask'][0])
        # # print(input['ipt_type_ids'][0])
        # print(input['target_ids'][0])
        # # print(input['tgt_mask'][0])
        # # print(input['tgt_type_ids'][0])
        # print(input['target_text'][0])
        
        break