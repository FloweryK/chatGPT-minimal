import pandas as pd
import torch
from torch.utils.data import Dataset
from constant import *


class KoreanQADataset(Dataset):
    def __init__(self, vocab, config):
        super().__init__()

        # load vocab
        self.vocab = vocab

        # load conversations
        self.data = self.load_data(config.path_data)
        self.len = len(self.data)
    
    def load_data(self, path_data):
        data = []
        df = pd.read_csv(path_data)
        for _, row in df.iterrows():
            question = row['Q']
            question_encode = [BOS] + self.vocab.EncodeAsIds(question) + [EOS]

            answer = row['A']
            answer_encode = [BOS] + self.vocab.EncodeAsIds(answer) + [EOS]

            data.append({
                'question': question, 
                'question_encode': question_encode,
                'answer': answer,
                'answer_encode': answer_encode,
            })
        
        return data
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        obj = self.data[index]
        
        return (
            torch.tensor(obj['question_encode']),
            torch.tensor(obj['answer_encode']),
        )