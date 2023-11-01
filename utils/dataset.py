import os
import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from utils.constant import *


class ChatDataset(Dataset):
    def __init__(self, path_data, path_prefix, n_vocab):
        super().__init__()

        # properties
        ## CHECKPOINT
        self.tokenizer = None
        self.data = None
        self.len = None

        # load data
        self.load_data(path_data, path_prefix, n_vocab)
    
    def load_data(self, path_data, path_prefix, n_vocab):
        # read csv
        df = pd.read_csv(path_data)

        # set data
        data = [(row['Q'], row['A']) for _, row in df.iterrows()]
        
        # set paths
        path_txt = path_prefix + '.txt'
        path_vocab = path_prefix + '.vocab'
        path_model = path_prefix + '.model'

        # train tokenizer
        with open(path_txt, 'w', encoding='utf8') as f:
            for text_q, text_a in data:
                f.write(' '.join([text_q]) + '\n')
                f.write(' '.join([text_a]) + '\n')
        
        spm.SentencePieceTrainer.train(
            input=path_txt,
            vocab_size=n_vocab,
            model_prefix=path_prefix,
            model_type='bpe',
            max_sentence_length=9999,
            pad_id=PAD,
            pad_piece='[PAD]',
            unk_id=UNK,
            unk_piece='[UNK]',
            bos_id=BOS,
            bos_piece='[BOS]',
            eos_id=EOS,
            eos_piece='[EOS]',
            user_defined_symbols=['[SEP]', '[CLS]', '[MASK]']
        )
        os.remove(path_txt)
        os.remove(path_vocab)

        # load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(path_model)

        # encode data
        self.data = [(
            [BOS] + self.tokenizer.EncodeAsIds(text_q) + [EOS],
            [BOS] + self.tokenizer.EncodeAsIds(text_a) + [EOS],
        ) for text_q, text_a in data]

        # set length
        self.len = len(self.data)
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        token_q = self.data[index][0]
        token_a = self.data[index][1]
        return torch.tensor(token_q), torch.tensor(token_a)

    @staticmethod
    def collate_fn(inputs):
        x_enc = []
        x_dec = []

        for data in inputs:
            x_enc.append(data[0])
            x_dec.append(data[1])

        x_enc = torch.nn.utils.rnn.pad_sequence(x_enc, batch_first=True, padding_value=PAD)
        x_dec = torch.nn.utils.rnn.pad_sequence(x_dec, batch_first=True, padding_value=PAD)
        
        return [x_enc, x_dec]