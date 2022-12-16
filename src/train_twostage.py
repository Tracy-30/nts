import argparse
import datetime
import time
import os
import shutil
import torch
import torch.backends.cudnn as cudnn
from utils import save, to_device, resume
from config import cfg, process_args

from data import fetch_dataset, make_data_loader
from models.summarizer import Two_Stage_Summarizer
from metrics.metrics import Metric, ROUGE
from logger import make_logger

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))

args = vars(parser.parse_args())
process_args(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['cuda_device'])

def main():
    for _ in range(cfg['num_experiments']):
        cfg['model_tag'] = '0_{}_{}'.format(cfg['data_name'],cfg['model_name'])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()

def runExperiment(): 
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])

    # load_dataset
    model_name = cfg['model_name']

    dataset = fetch_dataset(cfg['data_name'])
    data_loader = make_data_loader(dataset, model_name)
    print(f"train_data:{len(data_loader['train'])} test_data:{len(data_loader['test'])}")

    # load_model
    model = Two_Stage_Summarizer(**cfg[model_name]['param'])

    # load_metric
    metric = Metric({'train': ['Model_Loss', 'S1_Loss', 'S2_Loss'], 
                     'test': ['Rouge-1', 'Rouge-2', 'Rouge-L', 'Rouge-Avg']})

    # resume training 
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            model.load_state_dict(result['model_state_dict'])
            model.optimizer.load_state_dict(result['optimizer_state_dict'])
            model.scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            logger = make_logger('{}/runs/train_{}'.format(cfg['save_path'], cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('{}/runs/train_{}'.format(cfg['save_path'], cfg['model_tag']))

    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))

    print("Start Training")
    for epoch in range(last_epoch, cfg[model_name]['num_epochs'] + 1):
        train_twostage(data_loader, model, metric, logger, epoch)
        print('Epoch {} Training Totally Finish! Start Test Epoch {}'.format(epoch,epoch))
        test_twostage(data_loader['test'], model, metric, logger, epoch, split='test')

        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict,
                  'optimizer_state_dict': model.optimizer.state_dict(), 'scheduler_state_dict': model.scheduler.state_dict(), 'logger': logger}
        save(result, '{}/model/{}_checkpoint.pt'.format(cfg['save_path'],cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('{}/model/{}_checkpoint.pt'.format(cfg['save_path'], cfg['model_tag']),
                        '{}/model/{}_best.pt'.format(cfg['save_path'], cfg['model_tag']))
        logger.reset()
    return

def train_twostage(data_loaders, model, metric, logger, epoch):
    logger.safe(True)
    model.train()

    start_time = time.time()
    for i, input in enumerate(data_loaders['train']):
        input = to_device(input, cfg['device'])
        input_size = input['input_ids'].size(0)

        train_output = model(input, train=True) # {'model_loss', 'stage1_loss', 'stage2_loss'}

        evaluation = metric.evaluate(metric.metric_name['train'], input, train_output)
        logger.append(evaluation, 'train', n=input_size)

        # Update Scheduler
        if cfg[cfg['model_name']]['scheduler_name'] == 'ReduceLROnPlateau' and (not cfg[cfg['model_name']]['warm_up']):
            model.scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            model.scheduler.step()

        if i % cfg['info_interval'] == 0:  # print training info
            _time = (time.time() - start_time) / (i + 1)
            lr = model.optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loaders['train']) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loaders['train'])))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loaders['train'])),
                             'Learning rate: {:.8f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))

        if i % cfg['eval_interval'] == 0:
            print("Validate model during training: ")
            test_twostage(data_loaders['test'], model, metric, logger, epoch, split='validation')

            # # Update Scheduler
            # if cfg[cfg['model_name']]['scheduler_name'] == 'ReduceLROnPlateau' and (not cfg[cfg['model_name']]['warm_up']):
            #     model.scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
            # else:
            #     model.scheduler.step()

    logger.safe(False)
    return 

def test_twostage(data_loader, model, metric, logger, epoch, split='test'):
    if split == 'test':
        logger.safe(True)
        model.eval()
    with torch.no_grad():
        for i, input in enumerate(data_loader): # bs always 1
            if split == 'validation' and i > 0: break

            input = to_device(input, cfg['device'])
            input_size = input['input_ids'].size(0)

            pred_sentence = model(input, train=False)

            # test_output = model.train_one_batch(input,train=False)
            test_output = {}

            test_output['rouge_1'], test_output['rouge_2'], test_output['rouge_L'], test_output['rouge_avg']\
                 = ROUGE(pred_sentence,input['target_text'][0])

            evaluation = metric.evaluate(metric.metric_name['test'], input, test_output)
            logger.append(evaluation, split, n=input_size)

            print(i,':  ', pred_sentence)
            print(input['target_text'][0])

        info = {'info': ['Model: {}'.format(cfg['model_tag']), '{} Epoch: {}'.format(split, epoch)]}        
        logger.append(info, split, mean=False)
        print(logger.write(split, metric.metric_name['test']))
    if split == 'test':
        model.train()
        logger.safe(False)
    return

if __name__ == "__main__":
    main()
