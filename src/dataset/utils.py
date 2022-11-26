import pickle
from config import cfg
import errno
import os
import torch
import numpy as np


def check_exists(path): # check if the path exists
    return os.path.exists(path)

def makedir_exist_ok(path): # make a directory path
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


class TextInstance(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class TextFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def abst2stns(abstract):
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(cfg['SENTENCE_START'], cur)
      end_p = abstract.index(cfg['SENTENCE_END'], start_p + 1)
      cur = end_p + len(cfg['SENTENCE_END'])
      sents.append(abstract[start_p+len(cfg['SENTENCE_START']):end_p])
    except ValueError as e: # no more sentences
      return sents


def save(input, path, mode='torch'): # save data
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'): # load data
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return