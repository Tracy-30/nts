from torch.utils.data import Dataset
from dataset.utils import TextInstance, TextFeatures, abst2stns, check_exists, save, load
from tensorflow.core.example import example_pb2

from config import cfg
import os
import glob
import struct
from tqdm import tqdm


'''CNN/Daily Mail Dataset'''
class CNNDM(Dataset):
    def __init__(self, root, split, tokenizer):
        '''
        root: root PATH
        split: train/test
        tokenizer: BERT tokenizer
        '''
        self.root = os.path.expanduser(root)
        self.split = split

        if not check_exists(self.processed_folder):
            self.process()

        self.dataset = load(os.path.join(self.processed_folder, '{}.pickle'.format(self.split)), mode='pickle')
        self.total_num = len(self.dataset[0])

        self.bert_tokenizer = tokenizer

    def __getitem__(self, idx):
        bert_tokenized_feat = {}

        bert_tokenized_feat['ipt_feat']  = self.preprocess(self.dataset[0][idx], is_bert=True) # BERT article features  
        bert_tokenized_feat['tgt_feat']  = self.preprocess(self.dataset[1][idx], is_bert=True) # BERT summary features

        bert_tokenized_feat['tgt_txt']   = self.dataset[1][idx].text_a

        # if config.pointer_gen:
        #     # TODO: NEED TO UPDATE 
        #     item["input_ext_vocab_batch"], item["article_oovs"] = self.process_input(item["input_txt"])
        #     item["target_ext_vocab_batch"] = self.process_target(item["target_txt"], item["article_oovs"])
        #     item['target_ptr'], item['target_gate'] = self.create_ptr_and_gate(item["input_batch"],item["target_batch"],item["input_txt"],item["target_txt"])

        return bert_tokenized_feat

    def __len__(self):
        return self.total_num

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        train_artic, test_artic, val_artic, train_summ, test_summ, val_summ = self.make_data()

        save((train_artic, train_summ), os.path.join(self.processed_folder, 'train.pickle'), mode='pickle')
        save((test_artic, test_summ), os.path.join(self.processed_folder, 'test.pickle'), mode='pickle')
        save((val_artic, val_summ), os.path.join(self.processed_folder, 'val.pickle'), mode='pickle')

        return


    def make_data(self):
        train_artic, test_artic, val_artic = [], [], []
        train_summ, test_summ, val_summ = [], [], []

        unique_id = 0

        while True:
            filelist = glob.glob(f'{self.raw_folder}/*') # get the list of datafiles
            assert filelist, ('Error: Empty filelist at %s' % f'{self.raw_folder}/*') # check filelist isn't empty
            filelist = sorted(filelist)
            for f in tqdm(filelist):
                split = 'train' if 'train' in f else 'val' if 'val' in f else 'test'
                reader = open(f, 'rb')
                while True:
                    len_bytes = reader.read(8)
                    if not len_bytes: break # finished reading this file
                    str_len = struct.unpack('q', len_bytes)[0]
                    example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                    e = example_pb2.Example.FromString(example_str) 
                    
                    article_text = e.features.feature['article'].bytes_list.value[0]
                    abstract_text = e.features.feature['abstract'].bytes_list.value[0]

                    abstract_text = ' '.join([sent.strip() for sent in abst2stns(abstract_text.decode())])
                    
                    artic_inst = TextInstance(unique_id=unique_id,
                                     text_a=article_text.decode().strip(),
                                     text_b=None)
                    summ_inst =  TextInstance(unique_id=unique_id,
                                     text_a=abstract_text,
                                     text_b=None)

                    if split == 'train':
                        train_artic.append(artic_inst)
                        train_summ.append(summ_inst)
                    elif split == 'test':
                        test_artic.append(artic_inst)
                        test_summ.append(summ_inst)
                    elif split == 'val':
                        val_artic.append(artic_inst)
                        val_summ.append(summ_inst)

                    unique_id += 1
            
            return train_artic, test_artic, val_artic, train_summ, test_summ, val_summ


    def preprocess(self, text_instance, seq_length=cfg['max_seq_length'], is_bert=False):
        tokens_a = self.bert_tokenizer.tokenize(text_instance.text_a)
        tokens_b = None
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = [] # equals raw text tokens 
        input_type_ids = [] # equals segments_ids
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens) # WordPiece embedding rep

        if is_bert:
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            return TextFeatures(
                    unique_id=text_instance.unique_id,
                    tokens=tokens, # raw text tokens
                    input_ids=input_ids, # WordPiece tokens
                    input_mask=input_mask, # mask tokens for later
                    input_type_ids=input_type_ids) # segments_ids
        else:
            while len(input_ids) < seq_length:
                input_ids.append(0)
            return input_ids # WordPiece represetnation

    