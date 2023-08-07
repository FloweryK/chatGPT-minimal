import re
from dataset.base import ChatDatasetBase


def is_chat(line):
    pattern = r"^\d{4}년 \d{1,2}월 \d{1,2}일 (?:오전|오후) \d{1,2}:\d{2},"
    return bool(re.search(pattern, line))

def extract_chat(line):
    # extract date
    pattern = r"^\d{4}년 \d{1,2}월 \d{1,2}일 (?:오전|오후) \d{1,2}:\d{2},"
    date = re.findall(pattern, line)[0][:-1]
    line = re.sub(pattern, '', line)[1:-1]

    # extract speaker and text
    pattern = r".+ : "
    speaker = re.findall(pattern, line)[0][:-3]
    text = re.sub(pattern, '', line)

    return speaker, date, text

def is_emoticon(text):
    return text == '이모티콘'

def is_picture(text):
    return text == '사진'


class KakaotalkMobileDataset(ChatDatasetBase):
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
                speaker, date, text = extract_chat(line)

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
                        'question_speaker_name': speaker_prev,
                    }
                    i_prev = i
                else:
                    self.data[i_prev]['text'].append(text)

                speaker_prev = speaker