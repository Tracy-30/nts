import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_layers import DecoderLayer, LayerNorm
from models.utils import _gen_bias_mask , _gen_timing_signal, _get_attn_subsequent_mask
from config import cfg

from pytorch_pretrained_bert import BertModel

class Encoder(nn.Module):
    """
    BERT Encoder Module
    """
    def __init__(self) -> None:
        super(Encoder, self).__init__()

        # BERT Pre-trained Encoder
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.embedding = self.encoder.embeddings

        self.eval_() # always in eval mode

    def forward(self, input, mask, token_type_ids):
        """
        input: [bs, T]
        output: [bs, T, embedding_size=768]
        """

        with torch.no_grad():
            encoder_outputs, _ = self.encoder(input, attention_mask=mask, token_type_ids=token_type_ids,
                                                                output_all_encoded_layers=False)
        return encoder_outputs
    
    def eval_(self):
        self.encoder.eval()
        self.embedding = self.embedding.eval()

    
class Decoder(nn.Module):
    """
    A Transformer Decoder module. 
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=cfg[cfg['data_name']]['decoder_max_length'], input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0):
        """
        num_layers: Number of attention layers
        num_heads: Number of attention heads
        total_key_depth: Size of last dimension of keys. Must be divisible by num_head
        total_value_depth: Size of last dimension of values. Must be divisible by num_head
        filter_size: Hidden size of the middle layer in FFN

        max_length: max sequence length

        input_dropout: Dropout after embedding
        layer_dropout: Dropout of each layer
        attention_dropout: Dropout after attention 
        relu_dropout: Dropout after relu in FFN
        """
        
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        self.mask = _get_attn_subsequent_mask(max_length) # mask to hide future

        params =(hidden_size, total_key_depth or hidden_size, total_value_depth or hidden_size, filter_size, 
                num_heads, _gen_bias_mask(max_length), layer_dropout, attention_dropout, relu_dropout)
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)

        self.dec = nn.Sequential(*[DecoderLayer(*params) for _ in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        """
        inputs: [bs, T, hidden_size]
        outputs: [bs, T, hidden_size]
        """

        mask_src, mask_trg = mask
        if mask_trg is not None:
            dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        else:
            dec_mask = None

        # dropout input
        x = self.input_dropout(inputs)
        
        # timing signal
        x = self.embedding_proj(x)
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # Project to hidden size
        encoder_output = self.embedding_proj(encoder_output) #.transpose

        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src,dec_mask)))

        # Layer normalization
        y = self.layer_norm(y)

        return y, attn_dist

class LOGIT_LAYER(nn.Module):
    def __init__(self, hidden_dim, vocab) -> None:
        super(LOGIT_LAYER, self).__init__()
        self.output_layer = nn.Linear(hidden_dim, vocab)

    def forward(self, x, attn_dist=None, enc_input=None):
        logit = self.output_layer(x)
        log_probs = F.log_softmax(logit, dim=-1)

        return log_probs

class COPY_LAYER(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, hidden_dim, vocab, context=True):
        super(COPY_LAYER, self).__init__()
        self.proj_vocab_dis = nn.Linear(hidden_dim, vocab)
        if context:
            self.p_gen_layer = nn.Linear(cfg[cfg['model_name']]['param']['hidden_size']+768, 1)
        else:
            self.p_gen_layer = nn.Linear(cfg[cfg['model_name']]['param']['hidden_size'], 1)
        self.act = nn.Sigmoid()

    def forward(self, x, attn_dist, enc_input, enc_output):
        """x: [bs, dec_len=100, hidden_dim]  attn_dist: [bs, dec_len=100, input_length=512]  enc_input: [bs, input_length=512]"""
        """enc_output: [bs, input_length, embedding_size]"""
        
        if attn_dist is None: # No Copy
            logit = self.proj_vocab_dis(x)
            log_probs = F.log_softmax(logit, dim=-1)
            return log_probs

        else: # Copy
            if enc_output is not None: # Use Context Vector
                context_vector = attn_dist.bmm(enc_output) # [bs, dec_len=100, embedding_size]
                mix = torch.cat((context_vector, x), dim=2) # [bs, dec_len=100, embedding_size + hidden_dim]
                p_gen = self.p_gen_layer(mix) # [bs, dec_len=100, 1]

            else: # Don't Use Context Vector
                p_gen = self.p_gen_layer(x)

            p_gen = self.act(p_gen) # [bs, dec_len=100, 1]
            
            logit = self.proj_vocab_dis(x) # [bs, dec_len=100, vocab_size]
            vocab_dist = F.softmax(logit, dim=2)    # [bs, dec_len=100, vocab_size]
            vocab_dist_ = p_gen * vocab_dist        # [bs, dec_len=100, vocab_size]

            # attn_dist[:,:,0], attn_dist[:,:,-1] = 0, 0 # Avoid [CLS] and [SEP] tokens

            attn_dist = F.softmax(attn_dist, dim=2) # [bs, dec_len=100, input_length]
            attn_dist = (1 - p_gen) * attn_dist     # [bs, dec_len=100, input_length]

            add_position = enc_input.unsqueeze(1).repeat(1,x.size(1),1) # Avoid [CLS] and [SEP] tokens

            logit = torch.log(vocab_dist_.scatter_add(2, add_position, attn_dist))
                    
            return logit
        

