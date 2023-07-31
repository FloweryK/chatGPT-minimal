import os
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from constant import *


class ChatDatasetBase(Dataset):
    def __init__(self, n_vocab, path_data, path_vocab):
        super().__init__()
        if path_vocab is None:
            path_vocab = 'src/vocab/tmp.model'
    
        # load and encode data
        self.data = {}
        self.vocab = spm.SentencePieceProcessor()
        self.load_data(path_data)
        self.load_vocab(n_vocab, path_vocab)
        self.encode_data()

        # filter answers with no question
        self.ids = [value['id'] for value in self.data.values() if value['question_id']]
        self.len = len(self.ids)
    
    def load_data(self):
        raise NotImplementedError()
    
    def load_vocab(self, n_vocab, path_vocab):
        if path_vocab and (not os.path.exists(path_vocab)):
            path_prefix = path_vocab[:-6] 
            path_txt = path_prefix + '.txt'

            # create a tmp txt file for sentencepiece training
            with open(path_txt, 'w', encoding='utf8') as f:
                for chat in self.data.values():
                    f.write(' '.join(chat['text']) + '\n')
            
            # train sentencepiece
            spm.SentencePieceTrainer.train(
                input=path_txt,
                vocab_size=n_vocab,
                model_prefix=path_prefix,
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
        self.vocab.Load(path_vocab)

    def encode_data(self):
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


def collate_fn(inputs):
    x_enc, x_dec = list(zip(*inputs))

    x_enc = torch.nn.utils.rnn.pad_sequence(x_enc, batch_first=True, padding_value=PAD)
    x_dec = torch.nn.utils.rnn.pad_sequence(x_dec, batch_first=True, padding_value=PAD)
    
    return [x_enc, x_dec]