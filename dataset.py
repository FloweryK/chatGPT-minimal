import json
from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import Dataset
from constant import *


class MovieCorpusDataset(Dataset):
    def __init__(self, vocab, path_data):
        super().__init__()

        # load vocab
        self.vocab = vocab

        # load converstaions
        self.path_data = path_data
        self.data = defaultdict()
        self.load_data()

        self.ids = [value['id'] for value in self.data.values() if value['question_id']]
        self.len = len(self.ids)
    
    def load_data(self):
        with open(self.path_data, 'r', encoding='iso-8859-1') as f:
            for line in f:
                # load line as json
                obj = json.loads(line)

                # make converstaion data
                conversation_id = obj['conversation_id']
                question_id = obj['reply-to']
                answer_id = obj['id']
                text = obj['text'].strip()
                text_encode = [BOS] + self.vocab.EncodeAsIds(text) + [EOS]

                self.data[answer_id] = {
                    'conversation_id': conversation_id,
                    'question_id': question_id,
                    'id': answer_id,
                    'text': text,
                    'encode': text_encode
                }
                
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        answer_id = self.ids[index]
        answer = self.data[answer_id]

        question_id = answer['question_id']
        question = self.data[question_id]

        return (
            torch.tensor(question['encode']),
            torch.tensor(answer['encode']),
        )


class KoreanQADataset(Dataset):
    def __init__(self, vocab, path_data):
        super().__init__()

        # load vocab
        self.vocab = vocab

        # load conversations
        self.path_data = path_data
        self.data = []
        self.load_data()
        
        self.len = len(self.data)
    
    def __len__(self):
        return self.len
    
    def load_data(self):
        df = pd.read_csv(self.path_data)
        for _, row in df.iterrows():
            question = row['Q']
            question_encode = [BOS] + self.vocab.EncodeAsIds(question) + [EOS]

            answer = row['A']
            answer_encode = [BOS] + self.vocab.EncodeAsIds(answer) + [EOS]

            self.data.append({
                'question': question, 
                'question_encode': question_encode,
                'answer': answer,
                'answer_encode': answer_encode,
            })
            
    def __getitem__(self, index):
        obj = self.data[index]
        
        return (
            torch.tensor(obj['question_encode']),
            torch.tensor(obj['answer_encode']),
        )



def collate_fn(inputs):
    x_enc, x_dec = list(zip(*inputs))

    x_enc = torch.nn.utils.rnn.pad_sequence(x_enc, batch_first=True, padding_value=PAD)
    x_dec = torch.nn.utils.rnn.pad_sequence(x_dec, batch_first=True, padding_value=PAD)
    
    return [x_enc, x_dec]