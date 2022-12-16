import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
from config import cfg
import torch.nn.functional as F


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

def to_device(input, device): # 
    output = recur(lambda x, y: x.to(y), input, device)
    return output

def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output

def process_dataset(dataset):
    cfg['data_size'] = {'train': len(dataset['train']), 'test': len(dataset['test'])}
    return

def process_control():
    data_shape = {'LIBRISPEECH':[1,128,128],'LIBRISPEECH_SMALL':[1,64,64],'CIFAR10':[3, 32, 32],'MNIST':[1, 32, 32]}
    cfg['data_shape'] = data_shape[cfg['data_name']]

    cfg['vqvae'] = {'hidden_size': 32, 'ch_mult': [1,1,2,4], 'num_res_block': 2, 'embedding_size': 512,
            'num_embedding': 16384, 'vq_commit': 0.25, 'resolution':128, 'z_channels':128, 'dropout': 0.0,
            'structure':'vqgan','ema':True,'speaker_embedding':False}

    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    cfg[model_name]['batch_size'] = {'train': 12, 'test': 6}
    cfg[model_name]['optimizer_name'] = 'Adam'
    cfg[model_name]['lr'] = 5.0e-06
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['weight_decay'] = 0
    cfg[model_name]['betas'] = [0.5,0.9]
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['scheduler_name'] = 'ReduceLROnPlateau'
    cfg[model_name]['factor'] = 0.85
    cfg[model_name]['patience'] = 1
    cfg[model_name]['threshold'] = 1.0e-4
    cfg[model_name]['min_lr'] = 1.0e-8
    cfg[model_name]['num_epochs'] = 50
    
def make_optimizer(model, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg[tag]['lr'],
                               weight_decay=cfg[tag]['weight_decay'], eps=cfg[tag]['eps'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR': #SGD
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs'], eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau': #Adam
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler

def resume(model_tag, load_tag='checkpoint', verbose=True):
    if os.path.exists('/scratch/yg172/output/model/{}_{}.pt'.format(model_tag, load_tag)):
        result = load('/scratch/yg172/output/model/{}_{}.pt'.format(model_tag, load_tag))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
        result = {'epoch': last_epoch, 'logger': logger}
    if verbose:
        print('Resume from {}'.format(result['epoch']))
    return result

def collate(input):
    for k in input:
        input[k] = torch.stack(input[k])
    return input

def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse
