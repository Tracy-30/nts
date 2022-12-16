import numpy as np
from tqdm import tqdm
from models.beam_omt import Translator
from rouge import Rouge
from config import cfg

from dataset.utils import to_device

def evaluate(model, data, model_name='trs', mode='valid', verbose=False):
    hyp_g, ref, r1, r2, rl, r_avg = [],[],[],[],[],[]
    t = Translator(model)
    rouge = Rouge()

    l, loss = [], None
    pbar = tqdm(enumerate(data), total=len(data))
    for j, batch in pbar:
        if cfg['device'] == 'cuda': 
            batch = to_device(batch, cfg['device'])
        
        loss = model.train_one_batch(batch, train=False)
        l.append(loss.item())
            
        if j <= 1: 
            if mode != 'test':
                sent_g = model.decoder_greedy(batch) # 1-decoder generation. for testing
            else:
                sent_g = model.eval_one_batch(batch) # 2-decoder generation.
            # sent_b, _ = t.translate_batch(batch) # beam search

            for i, sent in enumerate(sent_g):
                hyp_g.append(sent) 
                ref.append(batch["tgt_txt"][i])
                rouges = rouge.get_scores(sent,batch["tgt_txt"][i])[0] # (hyp, ref)
                r1_val,r2_val,rl_val = rouges['rouge-1']["f"], rouges['rouge-2']["f"], rouges['rouge-l']["f"]
                r1.append(r1_val)
                r2.append(r2_val)
                rl.append(rl_val)
                r_avg.append(np.mean([r1_val,r2_val,rl_val]))
        pbar.set_description("EVAL loss:{:.4f} r_avg:{:.2f}".format(np.mean(l),np.mean(r_avg)))
        if j > 1 and mode == "train": break

    if l: loss = np.mean(l)
    r_avg = np.mean(r_avg)
    r1 = np.mean(r1)
    r2 = np.mean(r2)
    rl = np.mean(rl)

    if verbose:
        print("\nEVAL loss: {:.4f} r_avg: [{:.2f}] r1: {:.2f} r2: {:.2f} rl: {:.2f}".format(loss, r_avg, r1, r2, rl))
        for hyp, gold in zip(hyp_g, ref):
            print("HYP: ")
            print(hyp)
            print("GOLD: ")
            print(gold)
    return loss, r_avg
