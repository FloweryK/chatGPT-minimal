import json
from dataset.base import ChatDatasetBase


class MovieCorpusDataset(ChatDatasetBase):
    def load_data(self, path_data):
        with open(path_data, 'r', encoding='iso-8859-1') as f:
            for line in f:
                # load line as json
                obj = json.loads(line)

                # make converstaion data
                text = obj['text'].strip()
                answer_id = obj['id']
                question_id = obj['reply-to']

                self.data[answer_id] = {
                    'id': answer_id,
                    'text': [text],
                    'question_id': question_id,
                }