import os
import json
import pickle
from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import Dataset
from constant import *


def save(data, path):
    # make proper directories if needed.
    path_split = list(filter(None, path.split('/')))
    if len(path_split) > 1:
        dir = '/'.join(path_split[:-1])
        path = '/'.join(path_split)

        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

    # if saving json, use json dump.
    if path[-4:] == 'json':
        with open(path, 'w', encoding='UTF8') as f:
            json.dump(data, f, ensure_ascii=False)
    # if not, use pickle dump.
    else:
        with open(path, 'wb') as f:
            pickle.dump(data, f)


def load(path):
    # if loading json, use json load
    if path[-4:] == 'json':
        with open(path, encoding='UTF8') as f:
            result = json.load(f)
    # if not, use pickle load.
    else:
        with open(path, 'rb') as f:
            result = pickle.load(f)

    return result


class MovieCorpusDataset(Dataset):
    def __init__(self, vocab, path_data):
        super().__init__()

        # load vocab
        self.vocab = vocab

        # load converstaions
        self.path_data = path_data
        self.data = defaultdict()
        self.n_seq = 0
        self.load_data()

        self.ids = [value['id'] for value in self.data.values() if value['question_id']]
        self.len = len(self.ids)
    
    def load_data(self):
        try:
            cache = load(PATH_CACHE)
            self.data = cache['data']
            self.n_seq = cache['n_seq']
            self.len = cache['len']
        except:
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
                    # text_encode = [BOS] + [EOS]

                    self.data[answer_id] = {
                        'conversation_id': conversation_id,
                        'question_id': question_id,
                        'id': answer_id,
                        'text': text,
                        'encode': text_encode
                    }
                    
                    self.n_seq = max(self.n_seq, len(text_encode))
                
            # cache
            cache = {
                'data': self.data,
                'n_seq': self.n_seq,
            }
            save(cache, PATH_CACHE)
        
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


class TestDataset(Dataset):
    def __init__(self, vocab, path_data):
        super().__init__()

        # load vocab
        self.vocab = vocab

        # load conversations
        self.path_data = path_data
        self.data = []
        self.n_seq = 0
        self.load_data()
        
        self.len = len(self.data)
    
    def __len__(self):
        return self.len
    
    def load_data(self):
        try:
            cache = load(PATH_CACHE)
            self.data = cache['data']
            self.n_seq = cache['n_seq']
        except:
            df = pd.read_csv(self.path_data)
            for _, row in df.iterrows():
                question = row['Q']
                question_encode = [BOS] + self.vocab.EncodeAsIds(question) + [EOS]
                # question_encode = [BOS] + [EOS]

                answer = row['A']
                answer_encode = [BOS] + self.vocab.EncodeAsIds(answer) + [EOS]

                self.data.append({
                    'question': question, 
                    'question_encode': question_encode,
                    'answer': answer,
                    'answer_encode': answer_encode,
                })
                self.n_seq = max(self.n_seq, len(question_encode), len(answer_encode))
            
            # cache
            cache = {
                'data': self.data,
                'n_seq': self.n_seq
            }
            save(cache, PATH_CACHE)
        
    def __getitem__(self, index):
        obj = self.data[index]
        
        return (
            torch.tensor(obj['question_encode']),
            torch.tensor(obj['answer_encode']),
        )



def collate_fn(inputs):
    x_enc, x_dec = list(zip(*inputs))

    x_enc = torch.nn.utils.rnn.pad_sequence(x_enc, batch_first=True, padding_value=0)
    x_dec = torch.nn.utils.rnn.pad_sequence(x_dec, batch_first=True, padding_value=0)
    
    return [x_enc, x_dec]