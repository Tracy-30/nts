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
    from models.summarizer import Summarizer
    model = Summarizer(is_draft=True, emb_dim=64, hidden_dim=32, hop=12, heads=4, depth=4, filter=4)
    # from models.decoders import Decoder, Generator

    # decoder = Decoder(embedding_size=32, hidden_size=32, num_layers=4, num_heads=4, total_key_depth=8, 
                                # total_value_depth=8,filter_size=8,)

