import torch
import numpy as np

from config import cfg

class Greedy_Search():
    """
    This class performs greedy decoding
    """
    def __init__(self, decoder, output_layer, embedding, tokenizer) -> None:

        self.bert_embedding = embedding
        self.decoder = decoder
        self.output_layer = output_layer

        self.bert_tokenizer = tokenizer
        pass

    def decode(self, encoder_output):
        """
        returns a list of tokenized decoded sentence (bs=1)
        """
        # Always start with the SOS
        input = torch.ones(1, 1).fill_(cfg['SOS_idx']).long()
        if cfg['device'] == 'cuda':
            input = input.cuda()
        input_mask = input.data.eq(cfg['PAD_idx']).unsqueeze(0)

        decoded_words = []
        for i in range(cfg[cfg['data_name']]['decoder_max_length']):

            embedding_input = self.bert_embedding(input) # Bert Encoding [bs=1, len, 768]
            decoder_output, attn_dist = self.decoder(embedding_input, encoder_output, (None, input_mask))

            if cfg[cfg['model_name']]['copy_mech']:
                logit = self.output_layer(decoder_output, None, None, None) # [1, len, vocab_size]
            else:
                logit = self.output_layer(decoder_output, None, None) # [1, len, vocab_size]
            
            _, y_preds = torch.max(logit[:, -1], dim=1)
         
            decoded_words.append(self.bert_tokenizer.convert_ids_to_tokens(y_preds.tolist())[0])

            y_preds = y_preds.data[0]
            next_word = torch.ones(1, 1).long().fill_(y_preds)

            if cfg['device']=='cuda':
                next_word = next_word.cuda()
                
            input = torch.cat([input, next_word], dim=1) 
            input_mask = input.data.eq(cfg['PAD_idx']).unsqueeze(1)

        return self.get_sentence(decoded_words)

    def get_sentence(self, decoded_words):
        if ('[SEP]' not in decoded_words) and ('[PAD]' not in decoded_words): 
            return decoded_words
        else:
            for i in range(len(decoded_words)):
                if decoded_words[i] == '[SEP]' or decoded_words[i] == '[PAD]': 
                    break
            if decoded_words[0] == '[CLS]':
                return decoded_words[1:i]
            else: 
                return decoded_words[:i]

