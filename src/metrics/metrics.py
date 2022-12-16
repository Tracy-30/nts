import numpy as np
import sys
import os
from config import cfg
sys.path.append(os.path.abspath(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from rouge import Rouge

score_metric = Rouge()

def ROUGE(pred, ref): 
    """pred: string , ref: string, bs always 1"""
    rouge_scores = score_metric.get_scores(pred, ref)[0]
    return rouge_scores['rouge-1']["f"]*100, rouge_scores['rouge-2']["f"]*100, rouge_scores['rouge-l']["f"]*100, \
        np.mean([rouge_scores['rouge-1']["f"]*100,rouge_scores['rouge-2']["f"]*100,rouge_scores['rouge-l']["f"]*100])
        

class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {
                       'Model_Loss': (lambda input, output: output['model_loss'].item()),
                       'S1_Loss': (lambda input, output: output['stage1_loss'].item()),
                       'S2_Loss': (lambda input, output: output['stage2_loss'].item()),
                       'Rouge-1': (lambda input, output: output['rouge_1']),
                       'Rouge-2': (lambda input, output: output['rouge_2']),
                       'Rouge-L': (lambda input, output: output['rouge_L']),
                       'Rouge-Avg': (lambda input, output: output['rouge_avg'])
                       }
        
    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['CNN_DAILYMAILS', 'CNN_DAILYMAILS_SMALL', 'TLDR_NEWS']:
            pivot = -float('inf')
            pivot_direction = 'up'
            pivot_name = 'ROUGE'
        else:
            raise ValueError('Not valid data name')
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return
