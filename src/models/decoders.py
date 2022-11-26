import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from utils.data import text_input2bert_input
from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask,  get_input_from_batch, get_output_from_batch
from utils import config
import random
from numpy import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class Decoder(nn.Module):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=config.max_enc_steps, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        self.mask = _get_attn_subsequent_mask(max_length) # mask to hide future

        params =(hidden_size, 
                total_key_depth or hidden_size,
                total_value_depth or hidden_size,
                filter_size, 
                num_heads, 
                _gen_bias_mask(max_length), # mandatory
                layer_dropout, 
                attention_dropout, 
                relu_dropout)
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)

        # input to decoder: tuple consisting of decoder inputs and encoder output
        self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        if mask_trg is not None:
            dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        else:
            dec_mask = None
        #Add input dropout
        x = self.input_dropout(inputs)
        # Project to hidden size
        x = self.embedding_proj(x)
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # Project to hidden size
        encoder_output = self.embedding_proj(encoder_output) #.transpose
        # Run decoder. Input: x, encoder_outputs, attention_weight, mask_src, dec_mask
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))

        # Final layer normalization
        y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)
        self.copyloss = nn.BCELoss()
        self.m = nn.Sigmoid()

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False, copy_gate=None, copy_ptr=None, mask_trg=None):

        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            p_gen = self.m(p_gen)

        logit = self.proj(x) # simple linear projection

        if config.pointer_gen:
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = p_gen * vocab_dist

            attn_dist_ = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - p_gen) * attn_dist_           
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) ## extend for all seq
            if(beam_search):
                enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)]*x.size(0),0) ## extend for all seq
            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))
            
            return logit
        else:
            return F.log_softmax(logit,dim=-1)

