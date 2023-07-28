import re
import torch
from torch.utils.data import Dataset
from constant import *


class KakaotalkDataset(Dataset):
    def __init__(self, vocab, config):
        super().__init__()

        # load vocab
        self.vocab = vocab
        
        # data
        self.data = self.load_data(config.path_data)
        self.ids = [value['chat_id'] for value in self.data.values() if value['reply_chat_id']]

        # filter target speaker
        self.ids = [chat_id for chat_id in self.ids if self.data[chat_id]['speaker_name'] == config.target_speaker]
        self.len = len(self.ids)
    
    def load_data(self, path_data):
        def is_date(line):
            pattern = '---------------'
            return line.startswith(pattern)

        def extract_date(line):
            pattern = '---------------'
            line = line.split(pattern)[1][1:-1]
            return '-'.join([match.zfill(2) for match in re.findall(r"\d+", line)])

        def is_chat(line):
            pattern = r"^\[(.*?)\].*?\[(.*?)\]"
            matches = re.findall(pattern, line)
            return bool(matches)

        def extract_chat(line):
            pattern = r"^\[(.*?)\].*?\[(.*?)\]"
            speaker, t = re.findall(pattern, line)[0]
            chat = re.sub(pattern, '', line)[1:-1]

            # convert t to proper format
            is_afternoon = t[:2] == '오후'
            hour, minute = t[3:].split(':')
            
            hour = int(hour) % 12 + int(is_afternoon) * 12
            hour = str(hour).zfill(2)
            t = f'{hour}:{minute}'

            return speaker, t, chat
    
        def is_emoticon(chat):
            return chat == '이모티콘'
        
        # make qa data
        data = {}
        with open(path_data, 'r', encoding="utf8") as f:
            i_prev = None
            speaker_prev = None
            speaker_ids = {}

            for i, line in enumerate(f):
                if i < 3:
                    continue

                if is_date(line):
                    date = extract_date(line)
                elif is_chat(line):
                    speaker, t, chat = extract_chat(line)

                    if is_emoticon(chat):
                        continue
                    
                    if speaker not in speaker_ids:
                        speaker_ids[speaker] = len(speaker_ids)

                    if (i_prev is None) or (speaker_prev != speaker):
                        data[i] = {
                            'chat_id': i,
                            'datetime': f'{date} {t}',
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


