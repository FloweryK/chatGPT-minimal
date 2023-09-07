import re
from dataset.base import ChatDatasetBase


def extract_chat_pc_windows(line):
    pattern = r"^\[(.*?)\].*?\[(.*?)\]"
    speaker, t = re.findall(pattern, line)[0]
    text = re.sub(pattern, '', line)[1:-1]
    return speaker, text

def extract_chat_pc_mac(line):
    line = ','.join(line.split(',')[1:])
    speaker = line.split(',')[0][1:-1]
    text = line.split(',')[1][1:-1]
    return speaker, text

def extract_chat_mobile_android(line):
    # extract date
    pattern = r"^\d{4}년 \d{1,2}월 \d{1,2}일 (?:오전|오후) \d{1,2}:\d{2},"
    date = re.findall(pattern, line)[0][:-1]
    line = re.sub(pattern, '', line)[1:-1]

    # extract speaker and text
    pattern = r".+ : "
    speaker = re.findall(pattern, line)[0][:-3]
    text = re.sub(pattern, '', line)
    return speaker, text

def extract_chat_mobile_ios(line):
    line = ','.join(line.split(',')[1:])
    speaker = line.split(':')[0].strip()
    text = ','.join(line.split(':')[1:]).strip()

    if (speaker[0] == '"') and (speaker[-1] == '"'):
        raise IndexError()

    return speaker, text


# general filtering logics
def is_emoticon(text):
    return text == '이모티콘'


def is_picture(text):
    return text == '사진'


class KakaotalkDataset(ChatDatasetBase):
    def __init__(self, path_data, path_vocab, n_vocab, is_augment, augment_topn, augment_threshold, speaker):
        # base initialization
        super().__init__(path_data, path_vocab, n_vocab, is_augment, augment_topn, augment_threshold)
        
        # speaker filtering
        if speaker is not None:
            self.ids = [i for i in self.ids if self.data[i]['speaker_name'] == speaker]
            self.len = len(self.ids)


    def load_data(self, path_data):
        for func in [extract_chat_pc_windows, extract_chat_pc_mac, extract_chat_mobile_android, extract_chat_mobile_ios]:
            with open(path_data, 'r', encoding="utf8") as f:
                tmp = {}
                i_prev = None
                speaker_prev = None

                for i, line in enumerate(f):
                    # extract chat
                    try:
                        speaker, text = func(line)
                    except IndexError:
                        # TMP exception for passing '\n' in lines
                        continue

                    if is_emoticon(text):
                        continue
                    if is_picture(text):
                        continue

                    # add chat into data
                    if (i_prev is None) or (speaker_prev != speaker):
                        tmp[i] = {
                            'id': i,
                            'speaker_name': speaker,
                            'text': [text],
                            'question_id': i_prev,
                            'question_speaker_name': speaker_prev,
                        }
                        i_prev = i
                    else:
                        tmp[i_prev]['text'].append(text)

                    speaker_prev = speaker
                
                if len(self.data) < len(tmp):
                    self.data = tmp