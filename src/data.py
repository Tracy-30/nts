from torch.utils.data import DataLoader
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

import torch
import numpy as np
from config import cfg
from dataset.cnn_dm import CNNDM

def fetch_dataset(data_name):
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = '/Users/tracy/Desktop/data/{}'.format(data_name)

    if data_name in ['CNN_DAILYMAILS']:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        dataset['train'] = eval('{}(root=root, split=\'train\', tokenizer=tokenizer)'.format('CNNDM'))
        dataset['test'] = eval('{}(root=root, split=\'test\', tokenizer=tokenizer)'.format('CNNDM'))
    else:
        raise ValueError('Not valid dataset name')

    print('data ready')
    return dataset

def input_collate(batch):
    output = {key: [] for key in batch[0].keys()}
    for b in batch:
        for key in b:
            output[key].append(b[key])
    
    ipt_feat = output['ipt_feat']
    ipt_ids_batch = torch.tensor([f.input_ids for f in ipt_feat], dtype=torch.long)
    ipt_mask_batch = torch.tensor([f.input_mask for f in ipt_feat], dtype=torch.long)
    artic_idx_batch = torch.zeros(ipt_ids_batch.size(), dtype=torch.long)

    tgt_feat = output['tgt_feat']
    tgt_ids_batch = torch.tensor([f.input_ids for f in tgt_feat], dtype=torch.long)
    tgt_mask_batch = torch.tensor([f.input_mask for f in tgt_feat], dtype=torch.long)
    summ_idx_batch = torch.zeros(tgt_ids_batch.size(), dtype=torch.long)

    target_batch = tgt_ids_batch.transpose(0, 1)

    data = {'ipt_ids':ipt_ids_batch, 'ipt_mask':ipt_mask_batch, 'ipt_artic_idx':artic_idx_batch,
            'tgt_batch':target_batch, 'tgt_mask':tgt_mask_batch, 'tgt_summ_idx':summ_idx_batch,
            'tgt_txt':output['tgt_txt'] }
    
    return data 

def make_data_loader(dataset, sampler=None):
    data_loader = {}
    for split in dataset:
        _batch_size = cfg['batch_size'][split] 
        _shuffle = cfg['shuffle'][split]

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
    data_loader = make_data_loader(dataset)
    
    for batch, input in enumerate(data_loader['train']):
        print(input.keys())

        print("data_piece_1:     ")
        print(input['ipt_ids'][0], input['ipt_ids'].shape)
    #     # print(input['ipt_mask'][0])
    #     # print(input['ipt_artic_idx'][0])
    #     # print(input['tgt_batch'][0])
    #     # print(input['tgt_mask'][0])
    #     # print(input['tgt_summ_idx'][0])
    #     # print(input['tgt_txt'][0])

        
    #     # print("data_piece_2:     ")
    #     # print(input['ipt_ids'][1])
    #     # print(input['ipt_mask'][1])
    #     # print(input['ipt_artic_idx'][1])
    #     # print(input['tgt_batch'][1])
    #     # print(input['tgt_mask'][1])
    #     # print(input['tgt_summ_idx'][1])
    #     # print(input['tgt_txt'][1])
        break