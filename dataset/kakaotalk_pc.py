import re
from dataset.base import ChatDatasetBase


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

def is_emoticon(text):
    return text == '이모티콘'

def is_picture(text):
    return text == '사진'


class KakaotalkPCDataset(ChatDatasetBase):
    def __init__(self, n_vocab, path_data, path_vocab, speaker=None):
        super().__init__(n_vocab, path_data, path_vocab)

        if speaker is not None:
            self.ids = [value['id'] for value in self.data.values() if value['speaker_name'] == speaker]
            self.len = len(self.ids)
            
    def load_data(self, path_data):
        with open(path_data, 'r', encoding="utf8") as f:
            i_prev = None
            speaker_prev = None

            for i, line in enumerate(f):
                if not is_chat(line):
                    continue

                # extract chat
                speaker, t, text = extract_chat(line)

                if is_emoticon(text):
                    continue
                if is_picture(text):
                    continue

                # add chat into data
                if (i_prev is None) or (speaker_prev != speaker):
                    self.data[i] = {
                        'id': i,
                        'speaker_name': speaker,
                        'text': [text],
                        'question_id': i_prev,
                        'question__spekaer_name': speaker_prev,
                    }
                    i_prev = i
                else:
                    self.data[i_prev]['text'].append(text)

                speaker_prev = speaker