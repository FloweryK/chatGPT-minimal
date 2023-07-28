import json
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from constant import *


class MovieCorpusDataset(Dataset):
    def __init__(self, vocab, config):
        super().__init__()

        # load vocab
        self.vocab = vocab

        # load converstaions
        self.data = self.load_data(config.path_data)
        self.ids = [value['id'] for value in self.data.values() if value['question_id']]
        self.len = len(self.ids)
    
    def load_data(self, path_data):
        data = defaultdict()

        with open(path_data, 'r', encoding='iso-8859-1') as f:
            for line in f:
                # load line as json
                obj = json.loads(line)

                # make converstaion data
                conversation_id = obj['conversation_id']
                question_id = obj['reply-to']
                answer_id = obj['id']
                text = obj['text'].strip()
                text_encode = [BOS] + self.vocab.EncodeAsIds(text) + [EOS]

                data[answer_id] = {
                    'conversation_id': conversation_id,
                    'question_id': question_id,
                    'id': answer_id,
                    'text': text,
                    'encode': text_encode
                }
        
        return data
                
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
