import torch

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

