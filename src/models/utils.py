import torch
import numpy as np
import math

from config import cfg

def get_input_from_batch(batch):
    enc_batch = batch["input_batch"].transpose(0,1)
    batch_size, _ = enc_batch.size()
    input_ids_batch = batch["input_ids_batch"] #.transpose(0,1)
    input_mask_batch = batch["input_mask_batch"] #.transpose(0,1)
    example_index_batch = batch["example_index_batch"] #.transpose(0,1)
    batch_size, _ = input_ids_batch.size()

    extra_zeros = None
    enc_batch_extend_vocab = None

    if cfg['pointer_gen']:
        enc_batch_extend_vocab = batch["input_ext_vocab_batch"].transpose(0,1)
        # max_art_oovs is the max over all the article oov list in the batch
        if batch["max_art_oovs"] > 0:
            extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]))

    coverage = None
    if cfg['is_coverage']:
        coverage = torch.zeros(enc_batch.size())

    if cfg['device'] == 'cuda':
        if enc_batch_extend_vocab is not None:
                enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    return input_ids_batch, input_mask_batch, example_index_batch, enc_batch_extend_vocab, extra_zeros, coverage

def get_output_from_batch(batch):
    dec_batch = batch["target_batch"].transpose(0,1)
    dec_mask_batch = batch["target_mask_batch"]
    dec_index_batch = batch["target_index_batch"]

    target_gate,target_ptr = None, None
    # if(cfg['pointer_gen']):
    #     target_gate = batch["target_gate"]
    #     target_ptr = batch["target_ptr"]
    #     target_batch = batch["target_ext_vocab_batch"].transpose(0,1)
    # else:
    #     target_batch = dec_batch
        
    # dec_lens_var = batch["target_lengths"]
    # dec_padding_mask = sequence_mask(dec_lens_var, max_len=max_dec_len).float()

    return dec_batch, dec_mask_batch, dec_index_batch, target_gate, target_ptr #, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)
    
    return torch_mask.unsqueeze(0).unsqueeze(1)

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = ( math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal =  signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def _get_attn_subsequent_mask(size):
    """
    Get an attention mask to avoid using the subsequent info.
    Args:
        size: int
    Returns:
        (`LongTensor`):
        * subsequent_mask `[1 x size x size]`
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if cfg['device'] == 'cuda':
        return subsequent_mask.cuda()
    else:
        return subsequent_mask

