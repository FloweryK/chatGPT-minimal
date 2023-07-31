import re
import torch
from torch.utils.data import Dataset
from constant import *


class KakaotalkMobileDataset(Dataset):
    def __init__(self, vocab, path_data):
        super().__init__()

        # load vocab
        self.vocab = vocab
        
        # data
        self.data = self.load_data(path_data)
        self.ids = [value['chat_id'] for value in self.data.values() if value['reply_chat_id']]
        self.len = len(self.ids)
    
    def load_data(self, path_data):
        def is_chat(line):
            pattern = r"^\d{4}년 \d{1,2}월 \d{1,2}일 (?:오전|오후) \d{1,2}:\d{2},"
            return bool(re.search(pattern, line))

        def extract_chat(line):
            # extract date
            pattern = r"^\d{4}년 \d{1,2}월 \d{1,2}일 (?:오전|오후) \d{1,2}:\d{2},"
            date = re.findall(pattern, line)[0][:-1]
            line = re.sub(pattern, '', line)[1:-1]

            # extract speaker and chat
            pattern = r".+ : "
            speaker = re.findall(pattern, line)[0][:-3]
            chat = re.sub(pattern, '', line)

            return speaker, date, chat

        def is_emoticon(chat):
            return chat == '이모티콘'
            
        # make qa data
        data = {}
        with open(path_data, 'r', encoding="utf8") as f:
            i_prev = None
            speaker_prev = None
            speaker_ids = {}

            for i, line in enumerate(f):
                if is_chat(line):
                    speaker, date, chat = extract_chat(line)

                    if is_emoticon(chat):
                        continue
                    
                    if speaker not in speaker_ids:
                        speaker_ids[speaker] = len(speaker_ids)

                    if (i_prev is None) or (speaker_prev != speaker):
                        data[i] = {
                            'chat_id': i,
                            'speaker_id': speaker_ids[speaker],
                            'speaker_name': speaker,
                            'reply_chat_id': i_prev,
                            'reply_speaker_id': speaker_ids[speaker_prev] if speaker_prev else None,
                            'reply_speaker_name': speaker_prev,
                            'text': [chat],
                            'text_encode': [BOS] + self.vocab.EncodeAsIds(chat) + [EOS]
                        }
                        i_prev = i
                    else:
                        data[i_prev]['text'].append(chat)
                        data[i_prev]['text_encode'] = data[i_prev]['text_encode'][:-1] + [SEP] + self.vocab.EncodeAsIds(chat) + [EOS]

                    speaker_prev = speaker
        
        return data
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        answer_id = self.ids[index]
        answer = self.data[answer_id]

        question_id = answer['reply_chat_id']
        question = self.data[question_id]

        return (
            torch.tensor(question['text_encode']),
            torch.tensor(answer['text_encode']),
        )