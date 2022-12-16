# if __name__ == "__main__":
#     from dataset.cnn_dm import CNNDM

#     root = '/Users/tracy/Desktop/data/CNN_DAILYMAILS'

#     cnn_dm_data = CNNDM(root=root,split='train',tokenizer=None)

#     print(len(cnn_dm_data.dataset[0]))
#     print("raw data article: ")
#     print(cnn_dm_data.dataset[0][0].text_a)

#     print("raw data summary: ")
#     print(cnn_dm_data.dataset[1][0].text_a)

#     print(cnn_dm_data.__getitem__(0))

if __name__ == "__main__":
    from models.summarizer import Two_Stage_Summarizer
    from data import fetch_dataset, make_data_loader
    from config import cfg
    from metrics.metrics import ROUGE
    from utils import to_device

    dataset = fetch_dataset(cfg['data_name'])
    data_loader = make_data_loader(dataset,cfg['model_name'])

    param = cfg[cfg['model_name']]['param']
    model = Two_Stage_Summarizer(**param)

    param['two_stage'] = True
    model_refine = Two_Stage_Summarizer(**param)

    for i, input in enumerate(data_loader['test']):
        input = to_device(input, cfg['device'])
        pred_sentence1 = model.test_one_batch(input, method=cfg['search_algorithm'])
        pred_sentence2 = model_refine.test_one_batch(input, method=cfg['search_algorithm'])

        print(pred_sentence1)

        print(pred_sentence2)

        
        print(input['tgt_txt'])


        break

    

    
    

