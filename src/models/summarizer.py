import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import BertTokenizerFast

from modules.two_stage import Encoder, Decoder, LOGIT_LAYER, COPY_LAYER
from modules.search import Greedy_Search

from config import cfg
from utils import check_exists, makedir_exist_ok, make_optimizer, make_scheduler
from modules.regularize import LabelSmoothing
import transformers

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['cuda_device'])

class Draft_Decoder(nn.Module):
    '''Stage-1 Process'''
    def __init__(self, decoder, embedding) -> None:
        super(Draft_Decoder, self).__init__()

        self.decoder = decoder
        self.bert_embedding = embedding

    def forward(self, decoder_input, encoder_output):
        bs = encoder_output.size(0)

        sos_symbol = torch.LongTensor([cfg['SOS_idx']] * bs).unsqueeze(1) # Always start with the SOS_SYMBOL

        if cfg['device'] == 'cuda': 
            sos_symbol = sos_symbol.cuda()

        decoder_input_shift = torch.cat((sos_symbol, decoder_input[:, :-1]), dim=1) # shift the decoder input
        mask_trg = decoder_input_shift.data.eq(cfg['PAD_idx']).unsqueeze(1) # decoder input mask

        embedded_decoder_input = self.bert_embedding(decoder_input_shift) # Embed into BERT Representations

        decoder_output, attn_dist = self.decoder(embedded_decoder_input, encoder_output, (None,mask_trg)) # decode

        return decoder_output, attn_dist


class Refine_Decoder(nn.Module):
    '''Stage-2 Process'''
    def __init__(self, encoder, decoder, max_dec_step) -> None:
        super(Refine_Decoder, self).__init__()

        self.bert_encoder = encoder
        self.decoder = decoder
        self.max_ref_decode_step = max_dec_step

    def forward(self, ipt_ids_batch, ipt_mask_batch, artic_idx_batch, encoder_output):

        logits, attns = [],[]

        for i in range(self.max_ref_decode_step):
            
            with torch.no_grad():
                # Mask the location of each token.
                ref_ipt_mask_batch = torch.clone(ipt_mask_batch)
                ref_ipt_mask_batch[:, i] = 0

                # Bi-Directional BERT Representation
                bi_context_vector = self.bert_encoder(ipt_ids_batch, ref_ipt_mask_batch, artic_idx_batch)

                if cfg['device'] == 'cuda': 
                    bi_context_vector = bi_context_vector.cuda()

            pred, attn_dist = self.decoder(bi_context_vector, encoder_output, (None,None))

            # Combine all single predicted tokens
            logits.append(pred[:, i:i+1, :])
            attns.append(attn_dist[:, i:i+1, :])

        return torch.cat(logits, dim=1), torch.cat(attns, dim=1)

class Two_Stage_Summarizer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attn_heads, total_key_depth, total_value_depth, filter_size, 
                is_eval=False, two_stage=True) -> None:
        super(Two_Stage_Summarizer, self).__init__()
        """
        total_key_depth: Size of last dimension of keys. Must be divisible by num_head
        total_value_depth: Size of last dimension of values. Must be divisible by num_head
        filter_size: Hidden size of the middle layer in FFN
        """

        self.model_name = cfg['model_name']
        self.two_stage = two_stage

        # BERT Tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token

        # BERT Pre-trained Encoder
        self.encoder = Encoder()
        # Transformer Decoder
        self.decoder = Decoder(embedding_size=768, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_attn_heads, 
                            total_key_depth=total_key_depth, total_value_depth=total_value_depth, filter_size=filter_size,
                            max_length=cfg[cfg['data_name']]['decoder_max_length'],input_dropout=cfg[self.model_name]['dropout_rate'], 
                            layer_dropout=cfg[self.model_name]['dropout_rate'], attention_dropout=cfg[self.model_name]['dropout_rate'], 
                            relu_dropout=cfg[self.model_name]['dropout_rate'])

        # Output Layer 
        if cfg[self.model_name]['copy_mech']: # copy pointer-generator
            self.output_layer = COPY_LAYER(hidden_size, cfg['vocab_size'])
        else:
            self.output_layer = LOGIT_LAYER(hidden_size, cfg['vocab_size'])

        if cfg[self.model_name]['label_smooth'] > 0: 
            self.criterion = LabelSmoothing(size=cfg['vocab_size'], padding_idx=cfg['PAD_idx'], 
                                                    smoothing=cfg[self.model_name]['label_smooth'])
        else:
            # Negative Log Likelihood Loss
            self.criterion = nn.NLLLoss(ignore_index = cfg['PAD_idx'])

        # Optimizer
        self.optimizer = make_optimizer(self, self.model_name)
        # Scheduler
        if cfg[self.model_name]['warm_up']:
            num_training_steps =  cfg[self.model_name]['num_epochs'] * (cfg[cfg['data_name']]['train_samples'] // cfg[self.model_name]['batch_size']['train'])
            self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer, 
                    num_warmup_steps=num_training_steps*cfg[self.model_name]['num_warmup_partial'], num_training_steps=num_training_steps)
        else: self.scheduler = make_scheduler(self.optimizer, self.model_name)

        if cfg['device'] == 'cuda':
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.output_layer = self.output_layer.cuda()
            self.criterion = self.criterion.cuda()

        # Two Decoding Modules
        self.draft_decoder = Draft_Decoder(self.decoder, self.encoder.embedding)
        self.refine_decoder = Refine_Decoder(self.encoder, self.decoder, cfg[cfg['data_name']]['decoder_max_length'])

        # Searching Algorithms for Inference
        
        self.greedy_search = Greedy_Search(self.decoder, self.output_layer, self.encoder.embedding, self.tokenizer)
        
        # self.beam_search = 

    def train_one_batch(self, batch, train=True):
        input_ids, input_mask, input_type_ids, _, _, _ = get_input_from_batch(batch)
        target_ids, target_mask, target_type_ids, _, _ = get_output_from_batch(batch)

        # self.encoder.eval_()

        if train:
            self.optimizer.zero_grad()

        # First Stage Training (Teacher Forcing)
        encoder_output = self.encoder(input_ids, input_mask, input_type_ids)
        decoder_output_1, attn_dist_1 = self.draft_decoder(target_ids, encoder_output)

        stage1_logit = self.output_layer(decoder_output_1,attn_dist=attn_dist_1, enc_input=input_ids)
        stage1_loss = self.criterion(stage1_logit.contiguous().view(-1, stage1_logit.size(-1)), target_ids.contiguous().view(-1))

        if self.two_stage:
            # Second Stage Training (Teacher Forcing)
            decoder_output_2, attn_dist_2 = self.refine_decoder(target_ids, target_mask, target_type_ids, encoder_output)
            stage2_logit = self.output_layer(decoder_output_2,attn_dist=attn_dist_2, enc_input=input_ids)
            stage2_loss = self.criterion(stage2_logit.contiguous().view(-1, stage2_logit.size(-1)), target_ids.contiguous().view(-1))
        
        else: stage2_loss = torch.tensor(0, device=cfg['device'])

        if cfg['world_size'] > 1:
            stage1_loss = stage1_loss.mean()
            stage2_loss = stage2_loss.mean() if self.two_stage else stage2_loss

        model_loss =  stage1_loss + stage2_loss 
        
        if train:
            model_loss.backward()
            self.optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1)

        return {'model_loss':model_loss, 'stage1_loss': stage1_loss, 'stage2_loss': stage2_loss}

    def test_one_batch(self, batch, method='greedy'): # Test Batch_Size = 1
        input_ids, input_mask, input_type_ids, _, _, _ = get_input_from_batch(batch)

        # self.encoder.eval_()
        encoder_output = self.encoder(input_ids, input_mask, input_type_ids)

        # Stage-1 Draft
        if method == 'greedy':
            draft_output = self.greedy_search.decode(encoder_output)
        if method == 'beam':
            draft_output = self.beam_search.decode(encoder_output)

        """draft_output: one piece of draft_summary_tokens in list format []"""

        if not self.two_stage:
            return ' '.join(draft_output)

        else: # Stage-2 Refine
            draft_input, draft_input_mask, draft_summ_batch = bert_tokenization(draft_output, self.tokenizer)
            
            model_output, attn_dist = self.refine_decoder(draft_input, draft_input_mask, draft_summ_batch, encoder_output)
            model_output = self.output_layer(model_output, attn_dist, input_ids) # [bs=1, dec_step=100, vocab_size]

            refine_output = self.refine_decode(model_output)

            return ' '.join(refine_output)

    def refine_decode(self, output_logit): # batch_size = 1
        """Greedy Bi-Directional Context Decode"""
        decoded_words, sent = [], []
        _, word_indice = torch.max(output_logit, dim = -1)
        decoded_words = self.tokenizer.convert_ids_to_tokens(word_indice.squeeze().tolist())

        if '[SEP]' not in decoded_words and '[PAD]' not in decoded_words: 
            return decoded_words

        else:
            for i in range(len(decoded_words)):
                if decoded_words[i] == '[SEP]' or decoded_words[i] == '[PAD]': 
                    break
            if decoded_words[0] == '[CLS]':
                return decoded_words[1:i]
            else:
                return decoded_words[:i]

    def forward(self, batch, train=True):
        if train:
            self.train()
            output = self.train_one_batch(batch,train=True)
            return output
        else:
            self.eval()
            pred_sent = self.test_one_batch(batch)
            return pred_sent

def get_input_from_batch(batch):
    input_ids = batch["input_ids"]  
    input_mask = batch["input_mask"]  
    input_type_ids = batch["input_type_ids"]  
    batch_size, _ = input_ids.size()

    extra_zeros = None
    enc_batch_extend_vocab = None

    # if cfg[cfg['model_name']]['copy_mech']:
    #     enc_batch_extend_vocab = batch["input_ext_vocab_batch"].transpose(0,1)
    #     # max_art_oovs is the max over all the article oov list in the batch
    #     if batch["max_art_oovs"] > 0:
    #         extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]))

    coverage = None
    # if cfg['is_coverage']:
    #     coverage = torch.zeros(enc_batch.size())

    if cfg['device'] == 'cuda':
        if enc_batch_extend_vocab is not None: enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None: extra_zeros = extra_zeros.cuda()
        if coverage is not None: coverage = coverage.cuda()

    return input_ids, input_mask, input_type_ids, enc_batch_extend_vocab, extra_zeros, coverage

def get_output_from_batch(batch):
    target_ids = batch["target_ids"]
    target_mask = batch["target_mask"]
    target_type_ids = batch["target_type_ids"]

    target_gate,target_ptr = None, None
    # if(cfg['copy_mech']):
    #     target_gate = batch["target_gate"]
    #     target_ptr = batch["target_ptr"]
    #     target_batch = batch["target_ext_vocab_batch"].transpose(0,1)
    # else:
    #     target_batch = dec_batch
        
    # dec_lens_var = batch["target_lengths"]
    # dec_padding_mask = sequence_mask(dec_lens_var, max_len=max_dec_len).float()

    return target_ids, target_mask, target_type_ids, target_gate, target_ptr #, dec_padding_mask, max_dec_len, dec_lens_var, target_batch


def bert_tokenization(token_seq, bert_tokenizer, seq_length=cfg[cfg['data_name']]['decoder_max_length']):
    """manual tokenization for draft output"""
    if len(token_seq) > seq_length - 2:
        token_seq = token_seq[:(seq_length - 2)]

    tokens = ["[CLS]"] + token_seq + ["[SEP]"]

    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens) # WordPiece embedding rep

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    pad = seq_length - len(input_ids)
    if pad > 0: # Pad up to the sequence length.
        input_ids += [cfg['PAD_idx']] * pad
        input_mask += [0] * pad

    ipt_ids = torch.tensor(input_ids, dtype=torch.long, device=cfg['device']).unsqueeze(0)
    ipt_mask = torch.tensor(input_mask, dtype=torch.long, device=cfg['device']).unsqueeze(0)
    input_type_ids = torch.zeros(ipt_ids.size(), dtype=torch.long, device=cfg['device'])

    return ipt_ids, ipt_mask, input_type_ids



    









        



