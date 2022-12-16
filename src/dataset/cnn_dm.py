import os
import csv
import torch
from numpy.random import default_rng


from config import cfg
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer
from utils import check_exists, makedir_exist_ok, save, load
from dataset.utils import tokenize_with_truncation

'''CNN/Daily Mail Dataset'''
class CNNDM(Dataset):
    def __init__(self, root, split, tokenizer="bert-base-uncased"):
        super(CNNDM, self).__init__()
        '''
        root: root PATH
        split: train/test
        tokenizer: BERT tokenizer
        '''
        self.root = os.path.expanduser(root)
        self.split = split

        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
        with open(self.raw_folder + f'/{split}.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            c = 0
            for row in csv_reader:
                if c >= 1: # avoid column name
                    target_text.append(row[1])

                    input_data = tokenize_with_truncation(input=row[0],tokenizer=self.bert_tokenizer,
                            truncated_size=cfg[cfg['data_name']]['encoder_max_length'],padding_idx=cfg['PAD_idx'],t=True)

                    target_data = tokenize_with_truncation(input=row[1],tokenizer=self.bert_tokenizer,
                            truncated_size=cfg[cfg['data_name']]['decoder_max_length'],padding_idx=cfg['PAD_idx'],t=True)

                    input_ids.append(input_data['input_ids'])
                    input_mask.append(input_data['input_mask'])
                    input_type_ids.append(input_data['input_type_ids'])

                    target_ids.append(target_data['input_ids'])
                    target_mask.append(target_data['input_mask'])
                    target_type_ids.append(target_data['input_type_ids'])
                c+=1
        return target_text, torch.stack(input_ids), torch.stack(input_mask), torch.stack(input_type_ids), torch.stack(target_ids), \
            torch.stack(target_mask), torch.stack(target_type_ids)


'''A Small Version CNN/Daily Mail Dataset'''
class CNNDM_SMALL(Dataset):
    def __init__(self, root, split, tokenizer="bert-base-uncased", train_size=10000, test_size=100):
        '''
        root: root PATH
        split: train/test
        tokenizer: BERT tokenizer
        '''
        self.root = os.path.expanduser(root)
        self.split = split

        self.train_size = train_size
        self.test_size = test_size

        self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer)

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

        total_sample = 287227 if split=='train' else 11490
        max_sample = self.train_size if split=='train' else self.test_size

        rng = default_rng()
        sample_idx = rng.choice(total_sample, size=max_sample, replace=False)

        with open(self.raw_folder + f'/{split}.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            c = 0
            for row in csv_reader:
                if c >= 1 and c in sample_idx: # avoid column name
                    target_text.append(row[1])

                    input_data = tokenize_with_truncation(input=row[0],tokenizer=self.bert_tokenizer,
                            truncated_size=cfg[cfg['data_name']]['encoder_max_length'],padding_idx=cfg['PAD_idx'],t=True)

                    target_data = tokenize_with_truncation(input=row[1],tokenizer=self.bert_tokenizer,
                            truncated_size=cfg[cfg['data_name']]['decoder_max_length'],padding_idx=cfg['PAD_idx'],t=True)

                    input_ids.append(input_data['input_ids'])
                    input_mask.append(input_data['input_mask'])
                    input_type_ids.append(input_data['input_type_ids'])

                    target_ids.append(target_data['input_ids'])
                    target_mask.append(target_data['input_mask'])
                    target_type_ids.append(target_data['input_type_ids'])
                c += 1

        return target_text, torch.stack(input_ids), torch.stack(input_mask), torch.stack(input_type_ids), torch.stack(target_ids), \
            torch.stack(target_mask), torch.stack(target_type_ids)

    