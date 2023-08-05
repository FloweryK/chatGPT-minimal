import pandas as pd
from constant import *
from dataset.base import ChatDatasetBase


class BasicDataset(ChatDatasetBase):
    def load_data(self, path_data):
        df = pd.read_csv(path_data)
        for i, row in df.iterrows():
            question = row['Q']
            answer = row['A']

            question_id = 2*i
            answer_id = 2*i+1
            
            self.data[question_id] = {
                'id': question_id,
                'text': [question],
                'question_id': None,
            }
            self.data[answer_id] = {
                'id': answer_id,
                'text': [answer],
                'question_id': question_id,
            }