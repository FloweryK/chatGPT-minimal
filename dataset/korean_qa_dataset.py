import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from constant import *


class KoreanQADataset(Dataset):
    def __init__(self, path_data, n_vocab):
        super().__init__()

        # load and encode data
        self.data = {}
        self.vocab = None
        self.load_data(path_data)
        self.load_vocab(n_vocab)
        self.encode_data()
        self.len = len(self.data)
    
    def load_data(self, path_data):
        self.data = []
        df = pd.read_csv(path_data)
        for _, row in df.iterrows():
            question = row['Q']
            answer = row['A']

            self.data.append({
                'question': question, 
                'answer': answer,
            })

    def load_vocab(self, n_vocab):
        # train sentencepiece
        with open('tmp.txt', 'w', encoding='utf8') as f:
            for chat in self.data:
                f.write(chat['question'] + '\n')
                f.write(chat['answer'] + '\n')
        
        spm.SentencePieceTrainer.train(
            input='tmp.txt',
            vocab_size=n_vocab,
            model_prefix='tmp',
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

        # load vocab
        self.vocab = spm.SentencePieceProcessor()
        self.vocab.Load('tmp.model')

    def encode_data(self):
        # encode
        for i, chat in enumerate(self.data):
            question_encode = [BOS] + self.vocab.EncodeAsIds(chat['question']) + [EOS]
            answer_encode = [BOS] + self.vocab.EncodeAsIds(chat['answer']) + [EOS]

            self.data[i]['question_encode'] = question_encode
            self.data[i]['answer_encode'] = answer_encode
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        obj = self.data[index]
        
        return (
            torch.tensor(obj['question_encode']),
            torch.tensor(obj['answer_encode']),
        )