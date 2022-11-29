import torch
import torch.nn as nn

import numpy as np
from models.utils import text_input2bert_input
import random

import pprint
pp = pprint.PrettyPrinter(indent=1)
import os

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertForMaskedLM

from models.decoders import Decoder, Generator
from config import cfg

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

class Summarizer(nn.Module):
    def __init__(self, is_draft, emb_dim, hidden_dim, hop, heads, depth, filter,
                    model_file_path=None, is_eval=False, load_optim=False):
        super(Summarizer, self).__init__()
        self.is_draft = is_draft
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if is_draft: self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else: self.encoder = BertForMaskedLM.from_pretrained('bert-base-uncased')

        self.encoder.eval() 
        self.embedding = self.encoder.embeddings

        self.decoder = Decoder(emb_dim, hidden_dim, num_layers=hop, num_heads=heads, 
                                total_key_depth=depth,total_value_depth=depth,
                                filter_size=filter)

        self.generator = Generator(hidden_dim,cfg['vocab_size'])

        self.criterion = nn.NLLLoss(ignore_index=cfg['PAD_idx'])
        self.criterion_ppl = nn.NLLLoss(ignore_index=cfg['PAD_idx'])
        
        
        self.embedding = self.embedding.eval()
        if is_eval:
            self.decoder = self.decoder.eval()
            self.generator = self.generator.eval()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg['lr'])
        
        if cfg['device'] == 'cuda':
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.generator = self.generator.cuda()
            self.criterion = self.criterion.cuda()
            self.embedding = self.embedding.cuda()

        self.model_dir = cfg['save_path']
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def train_one_batch(self, batch, train=True):
        ## pad and other stuff
        input_ids_batch, input_mask_batch, example_index_batch, enc_batch_extend_vocab, extra_zeros, _ = get_input_from_batch(batch)
        dec_batch, dec_mask_batch, dec_index_batch, copy_gate, copy_ptr = get_output_from_batch(batch)
        
        self.optimizer.zero_grad()

        # print('input: ', input_ids_batch.shape)
        with torch.no_grad():
            encoder_outputs, _ = self.encoder(input_ids_batch, token_type_ids=example_index_batch, 
                                        attention_mask=input_mask_batch, output_all_encoded_layers=False)
        # print('encoder_output: ', encoder_outputs.shape)

        # # Draft Decoder 
        sos_token = torch.LongTensor([cfg['SOS_idx']] * input_ids_batch.size(0)).unsqueeze(1)
        if cfg['device']=='cuda': sos_token = sos_token.cuda(device=0)

        dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1) # shift the decoder input (summary) by one step
        mask_trg = dec_batch_shift.data.eq(cfg['PAD_idx']).unsqueeze(1)

        pre_logit1, attn_dist1 = self.decoder(self.embedding(dec_batch_shift),encoder_outputs, (None,mask_trg))
        logit1 = self.generator(pre_logit1,attn_dist1,enc_batch_extend_vocab, extra_zeros, copy_gate=copy_gate, copy_ptr=copy_ptr, mask_trg= mask_trg)
        loss1 = self.criterion(logit1.contiguous().view(-1, logit1.size(-1)), dec_batch.contiguous().view(-1))

        # Refine Decoder - train using gold label TARGET
        'TODO: turn gold-target-text into BERT insertable representation'
        pre_logit2, attn_dist2 = self.generate_refinement_output(encoder_outputs, dec_batch, dec_index_batch, extra_zeros, dec_mask_batch)        
        logit2 = self.generator(pre_logit2,attn_dist2,enc_batch_extend_vocab, extra_zeros, copy_gate=copy_gate, copy_ptr=copy_ptr, mask_trg= None)
        loss2 = self.criterion(logit2.contiguous().view(-1, logit2.size(-1)), dec_batch.contiguous().view(-1))

        loss = loss1+loss2

        if train:
            loss.backward()
            self.optimizer.step()

        return loss

    def eval_one_batch(self, batch):
        draft_seq_batch = self.decoder_greedy(batch)

        input_ids_batch, input_mask_batch, example_index_batch, enc_batch_extend_vocab, extra_zeros, _ = get_input_from_batch(batch)
        dec_batch, dec_mask_batch, dec_index_batch, copy_gate, copy_ptr = get_output_from_batch(batch)

        with torch.no_grad():
            encoder_outputs, _ = self.encoder(input_ids_batch, token_type_ids=example_index_batch, 
                                        attention_mask=input_mask_batch, output_all_encoded_layers=False)

        d_seq_input_ids_batch, d_seq_input_mask_batch, d_seq_example_index_batch = text_input2bert_input(draft_seq_batch, self.tokenizer)
        pre_logit2, attn_dist2 = self.generate_refinement_output(encoder_outputs, d_seq_input_ids_batch, 
                                                d_seq_example_index_batch, extra_zeros, d_seq_input_mask_batch)

        decoded_words, sent = [], []
        for out, attn_dist in zip(pre_logit2, attn_dist2):
            prob = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros, copy_gate=copy_gate, copy_ptr=copy_ptr, mask_trg= None)
            _, next_word = torch.max(prob[:, -1], dim = 1)
            decoded_words.append(self.tokenizer.convert_ids_to_tokens(next_word.tolist()))

        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>' or e.strip() == '<PAD>': break
                else: st+= e + ' '
            sent.append(st)
        return sent

    def generate_refinement_output(self, encoder_outputs, input_ids_batch, example_index_batch, extra_zeros, input_mask_batch):
        # mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        # decoded_words = []
        logits, attns = [],[]

        for i in range(cfg['max_dec_step']):
            # print(i)
            with torch.no_grad():
                # Additionally mask the location of i. 
                context_input_mask_batch = []
                # print(context_input_mask_batch.shape) # (2,512) (batch_size, seq_len)
                for mask in input_mask_batch:
                    mask[i]=0
                    context_input_mask_batch.append(mask)

                context_input_mask_batch = torch.stack(context_input_mask_batch) #.cuda(device=0)
                    # self.embedding = self.embedding.cuda(device=0)
                context_vector, _ = self.encoder(input_ids_batch, token_type_ids=example_index_batch, attention_mask=context_input_mask_batch, output_all_encoded_layers=False)
                
                if cfg['device']=='cuda': context_vector = context_vector.cuda(device=0)
            # decoder input size == encoder output size == (batch_size, 512, 768)
            out, attn_dist = self.decoder(context_vector,encoder_outputs, (None,None))

            logits.append(out[:,i:i+1,:])
            attns.append(attn_dist[:,i:i+1,:])
        
        logits = torch.cat(logits, dim=1)
        attns = torch.cat(attns, dim=1)

        # print(logits.size(), attns.size())
        return logits, attns

    def decoder_greedy(self, batch):
        input_ids_batch, input_mask_batch, example_index_batch, enc_batch_extend_vocab, extra_zeros, _ = get_input_from_batch(batch)
        # mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        with torch.no_grad():
            encoder_outputs, _ = self.encoder(input_ids_batch, token_type_ids=enc_batch_extend_vocab, attention_mask=input_mask_batch, output_all_encoded_layers=False)

        ys = torch.ones(1, 1).fill_(cfg['SOS_idx']).long()
        if cfg['device']=='cuda': ys = ys.cuda()
        mask_trg = ys.data.eq(cfg['PAD_idx']).unsqueeze(1)
        decoded_words = []
        for i in range(cfg['max_dec_step']):
            out, attn_dist = self.decoder(self.embedding(ys),encoder_outputs, (None,mask_trg))
            prob = self.generator(out,attn_dist,enc_batch_extend_vocab, extra_zeros)
            _, next_word = torch.max(prob[:, -1], dim = 1)

            decoded_words.append(self.tokenizer.convert_ids_to_tokens(next_word.tolist()))

            next_word = next_word.data[0]
            if cfg['device']=='cuda':
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
            mask_trg = ys.data.eq(cfg['PAD_idx']).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>' or e == '<PAD>': break
                else: st+= e + ' '
            sent.append(st)
        return sent

    def save_model(self, loss, iter, r_avg):
        state = {
            'iter': iter,
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            #'optimizer': self.optimizer.state_dict(),
            'current_loss': loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}'.format(iter,loss,r_avg))
        self.best_path = model_save_path
        torch.save(state, model_save_path)


def get_input_from_batch(batch):
    input_ids_batch = batch["ipt_ids"] #.transpose(0,1)
    input_mask_batch = batch["ipt_mask"] #.transpose(0,1)
    example_index_batch = batch["ipt_artic_idx"] #.transpose(0,1)
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
        if enc_batch_extend_vocab is not None: enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None: extra_zeros = extra_zeros.cuda()
        if coverage is not None: coverage = coverage.cuda()

    return input_ids_batch, input_mask_batch, example_index_batch, enc_batch_extend_vocab, extra_zeros, coverage

def get_output_from_batch(batch):
    dec_batch = batch["tgt_batch"].transpose(0,1)
    dec_mask_batch = batch["tgt_mask"]
    dec_index_batch = batch["tgt_summ_idx"]

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
