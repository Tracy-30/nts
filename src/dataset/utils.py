# import pickle
from config import cfg
# import errno
# import os
import torch
# import numpy as np



def tokenize_with_truncation(input, tokenizer, truncated_size, padding_idx, t=False):
    """
    Use Right Truncation
    input: a string of sequence 'a list of tokens'
    """
    input = tokenizer.tokenize(input) # a string of sequence ['a', 'list', 'of', 'tokens']

    if len(input) > truncated_size - 2:
        input = input[:(truncated_size-2)]

    input = ['[CLS]'] + input + ['[SEP]']

    input_mask = [1] * len(input)
    input_ids = tokenizer.convert_tokens_to_ids(input)

    pad_len = truncated_size - len(input_mask)
    if pad_len > 0:
        input_mask += [0] * pad_len
        input_ids += [padding_idx] * pad_len

    input_type_ids = [0] * truncated_size

    if t:
        input_ids = torch.LongTensor(input_ids)
        input_mask = torch.LongTensor(input_mask)
        input_type_ids = torch.LongTensor(input_type_ids)

    return {'input_ids': input_ids, 'input_mask': input_mask, 'input_type_ids': input_type_ids}


def extend_vocab(input_token_seq, output_token_seq, vocab_size, tokenizer):
    """
    input: ['I', 'missisipi -> [UNK]', 'what'] -> 
    output: ['She', 'is', 'gooood -> [UNK]'] -> 
    """
    input_ids = []
    oov = []
    for token in input_token_seq:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == cfg['UNK_idx']:
            if token not in oov:
                oov.append(token)
            input_ids.append(vocab_size + oov.index(token))
        else:
            input_ids.append(token_id)

    output_ids = []
    for token in output_token_seq:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == cfg['UNK_idx']:
            if token in oov:
                output_ids.append(vocab_size + oov.index(token))
            else:
                output_ids.append(cfg['UNK_idx'])
        else:
            output_ids.append(token_id)

    return input_ids, output_ids, oov




    

    







