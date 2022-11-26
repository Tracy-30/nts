if __name__ == "__main__":
    from dataset.cnn_dm import CNNDM

    root = '/Users/tracy/Desktop/data/CNN_DAILYMAILS'

    cnn_dm_data = CNNDM(root=root,split='train',tokenizer=None)

    print(len(cnn_dm_data.dataset[0]))
    print("raw data article: ")
    print(cnn_dm_data.dataset[0][0].text_a)

    print("raw data summary: ")
    print(cnn_dm_data.dataset[1][0].text_a)

    print(cnn_dm_data.__getitem__(0))
    