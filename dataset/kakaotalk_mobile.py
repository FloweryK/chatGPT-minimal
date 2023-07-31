import re
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from constant import *


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


class KakaotalkMobileDataset(Dataset):
    def __init__(self, path_data, n_vocab):
        super().__init__()

        # load and encode data
        self.data = {}
        self.vocab = None
        self.load_data(path_data)
        self.load_vocab(n_vocab)
        self.encode_data()

        # filter answers with no question
        self.ids = [value['id'] for value in self.data.values() if value['question_id']]
        self.len = len(self.ids)
    
    def load_data(self, path_data):
        with open(path_data, 'r', encoding="utf8") as f:
            i_prev = None
            speaker_prev = None

            for i, line in enumerate(f):
                # check chat validity
                if not is_chat(line):
                    continue

                # extract chat
                speaker, date, text = extract_chat(line)

                # check text validity
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
    
    def load_vocab(self, n_vocab):
        # train sentencepiece
        with open('tmp.txt', 'w', encoding='utf8') as f:
            for chat in self.data.values():
                f.write(' '.join(chat['text']) + '\n')
        
        spm.SentencePieceTrainer.train(
            input='tmp.txt',
            vocab_size=n_vocab,
            model_prefix='tmp',
            model_type='bpe',
            max_sentence_length=9999,
            pad_id=PAD,
            pad_piece='[PAD]',
            unk_id=UNK,
            unk_piece='[UNK]',
            bos_id=BOS,
            bos_piece='[BOS]',
            eos_id=EOS,
            eos_piece='[EOS]',
            user_defined_symbols=['[SEP]', '[CLS]', '[MASK]']
        )

        # load vocab
        self.vocab = spm.SentencePieceProcessor()
        self.vocab.Load('tmp.model')
    
    def encode_data(self):
        # encode
        for chat_id in self.data:
            text = self.data[chat_id]['text']
            text_encode = [BOS]
            for t in text:
                text_encode.extend(self.vocab.EncodeAsIds(t) + [SEP])
            text_encode[-1] = EOS
            self.data[chat_id]['text_encode'] = text_encode
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        answer_id = self.ids[index]
        answer = self.data[answer_id]

        question_id = answer['question_id']
        question = self.data[question_id]

        return (
            torch.tensor(question['text_encode']),
            torch.tensor(answer['text_encode']),
        )