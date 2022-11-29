from models.summarizer import Summarizer
from evaluate import evaluate
from config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import os
import time 
import numpy as np 
from data import fetch_dataset, make_data_loader

from pytorch_pretrained_bert.tokenization import BertTokenizer

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_draft():
    dataset = fetch_dataset(cfg['data_name'])
    data_loader = make_data_loader(dataset)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # if(config.test):
    #     print("Test model",config.model)
    #     model = Transformer(model_file_path=config.save_path,is_eval=True)
    #     evaluate(model,data_loader_test,model_name=config.model,ty='test')
    #     exit(0)

    model = Summarizer(is_draft=True, emb_dim=768, hidden_dim=32, hop=4, heads=4, depth=4, filter=4)
    print("TRAINABLE PARAMETERS",count_parameters(model))
    print("Device: ", cfg['device'])

    best_rouge = 0 
    cnt = 0
    eval_iterval = 500
    for e in range(cfg['epochs']):
        # model.train()
        print("Epoch", e)
        l = []
        pbar = tqdm(enumerate(data_loader['train']),total=len(data_loader['train']))
        for i, d in pbar:
            loss = model.train_one_batch(d)
            
            l.append(loss.item())
            pbar.set_description("TRAIN loss:{:.4f}".format(np.mean(l)))

            if i % eval_iterval == 0:
                model.eval()
                loss,r_avg = evaluate(model, data_loader['test'], model_name=cfg['model_name'], ty="train")
                # each epoch is long,so just do early stopping here. 
                if r_avg > best_rouge:
                    best_rouge = r_avg
                    cnt = 0
                    model.save_model(loss,e,r_avg)
                else: 
                    cnt += 1
                if cnt > 20: break
                model.train()
        # Test Epoch
        model.eval()
        loss,r_avg = evaluate(model,data_loader['test'],model_name=cfg['model_name'],ty="valid")

if __name__ == "__main__":
    # train_draft()
    dataset = fetch_dataset(cfg['data_name'])
    data_loader = make_data_loader(dataset)

    model = Summarizer(is_draft=True, emb_dim=768, hidden_dim=32, hop=4, heads=4, depth=4, filter=4)
    loss,r_avg = evaluate(model,data_loader['test'],model_name=cfg['model_name'],ty="train")