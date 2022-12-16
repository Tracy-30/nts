# Pre-training Two-stage Text Summarization

This is a PyTorch implementation of abstractive text summarization from the paper *Pretrain-Based Natural Language Generation for Text Summarization* (<https://arxiv.org/abs/1902.09243>)

We focused our dataset on CNN/Daily Mail (<https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail>). Due to time and computational resource limit, we also provide a smaller version of CNN/Daily Mail dataset, which contains randomly sampled 10000/100 (train/test) article-summary data piece from the original dataset.

## Training

To train by yourself, simply follow the steps:

### Step-1

Download the full train/test dataset of CNN/Daily Mails from <https://drive.google.com/drive/folders/1UNIPA4ROyhfz6r2xj__0eAB0u2c8ipwZ?usp=sharing>.

(This dataset is a pre-tokenized version of the raw CNN/Daily Mails in csv format produced from <https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail>)

Put the downloaded raw folder under data/CNN_DAILYMAILS or data/CNN_DAILYMAILS_SMALL

### Step-2

In config.py, change the path to the local path of config.yml

In config.yml, change data_path to the local path of /data folder, and change save_path to the local path

### Step-3

run following

```{bash}
python3 train_twostage.py --model_name two_stage_summarizer --data_name CNN_DAILYMAILS_SMALL --cuda_device 0 --eval_interval 500
```

More specific model parameters can be tuned in config.yml

## Inference

To make model inference, we also provide a infer.ipynb file to show some examples.
