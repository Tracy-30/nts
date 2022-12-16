import os
import csv
import torch
from numpy.random import default_rng


from config import cfg
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from utils import check_exists, makedir_exist_ok, save, load


'''A Small Version CNN/Daily Mail Dataset'''
class TLDR_NEWS(Dataset):
    def __init__(self, root, split, tokenizer="bert-base-uncased", train_size=7000, test_size=200):
        '''
        root: root PATH
        split: train/test
        tokenizer: BERT tokenizer
        '''
        self.root = os.path.expanduser(root)
        self.split = split

        self.train_size = train_size
        self.test_size = test_size

        self.bert_tokenizer = BertTokenizerFast.from_pretrained(tokenizer)
        self.bert_tokenizer.bos_token = self.bert_tokenizer.cls_token
        self.bert_tokenizer.eos_token = self.bert_tokenizer.sep_token

        if not check_exists(self.processed_folder):
            self.process()

        self.dataset = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)), mode='torch')
        self.tgt_txt = load(os.path.join(self.processed_folder, '{}_tgt.pickle'.format(self.split)), mode='pickle')


    def __getitem__(self, idx):
        bert_tokenized_feat = { 'input_ids':self.dataset[0][idx], 
                                'input_mask':self.dataset[1][idx], 
                                'input_type_ids':self.dataset[2][idx],
                                'target_ids':self.dataset[3][idx], 
                                'target_mask':self.dataset[4][idx], 
                                'target_type_ids':self.dataset[5][idx]}
        bert_tokenized_feat['target_text'] = self.tgt_txt[idx]

        # input should have length 512
        # output should have length 100

        return bert_tokenized_feat

    def __len__(self):
        return len(self.dataset[0])

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        makedir_exist_ok(self.processed_folder)
        train_data = self.make_data(split='train')
        test_data = self.make_data(split='test')

        save(train_data[1:7], os.path.join(self.processed_folder, 'train.pt'), mode='torch')
        save(test_data[1:7], os.path.join(self.processed_folder, 'test.pt'), mode='torch')

        save(train_data[0], os.path.join(self.processed_folder, 'train_tgt.pickle'), mode='pickle')
        save(test_data[0], os.path.join(self.processed_folder, 'test_tgt.pickle'), mode='pickle')

        return

    def make_data(self, split):
        input_ids, input_mask, input_type_ids = [], [], []
        target_ids, target_mask, target_type_ids = [], [], []

        target_text = []

        total_sample = 7138 if split=='train' else 794
        max_sample = self.train_size if split=='train' else self.test_size

        rng = default_rng()
        sample_idx = rng.choice(total_sample, size=max_sample, replace=False)

        with open(self.raw_folder + f'/{split}.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            c = 0
            for row in csv_reader:
                if c >= 1 and c in sample_idx: # avoid column name
                    target_text.append(row[1]) # headline 

                    input_data = self.bert_tokenizer(row[2], padding="max_length", truncation=True, max_length = cfg[cfg['data_name']]['encoder_max_length'])
                    target_data = self.bert_tokenizer(row[1], padding="max_length", truncation=True, max_length = cfg[cfg['data_name']]['decoder_max_length'])

                    input_ids.append(torch.LongTensor(input_data['input_ids']))
                    input_mask.append(torch.LongTensor(input_data['attention_mask']))
                    input_type_ids.append(torch.LongTensor(input_data['token_type_ids']))

                    target_ids.append(torch.LongTensor(target_data['input_ids']))
                    target_mask.append(torch.LongTensor(target_data['attention_mask']))
                    target_type_ids.append(torch.LongTensor(target_data['token_type_ids']))
                c += 1
        return target_text, torch.stack(input_ids), torch.stack(input_mask), torch.stack(input_type_ids), torch.stack(target_ids), \
            torch.stack(target_mask), torch.stack(target_type_ids)