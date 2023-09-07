import os
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from constant import *


class ChatDatasetBase(Dataset):
    def __init__(self, path_data, path_prefix, n_vocab, is_augment, augment_topn, augment_threshold):
        super().__init__()

        # load and encode data
        self.data = {}
        self.vocab = spm.SentencePieceProcessor()
        self.load_data(path_data)
        self.load_vocab(path_prefix, n_vocab)
        self.encode_data()
        
        # data augmentation
        if is_augment:
            self.augment_data(augment_topn, augment_threshold)

        # filter answers with no question
        self.ids = [value['id'] for value in self.data.values() if value['question_id']]
        self.len = len(self.ids)
    
    def load_data(self):
        raise NotImplementedError()
    
    def load_vocab(self, path_prefix, n_vocab):
        # set paths
        path_txt = path_prefix + '.txt'
        path_tmp = path_prefix + '.vocab'
        path_vocab = path_prefix + '.model'

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

        # delete unnecessary files
        os.remove(path_txt)
        os.remove(path_tmp)

        # load vocab
        self.vocab.Load(path_vocab)

    def encode_data(self):
        for chat_id in self.data:
            # extract text
            text = self.data[chat_id]['text']

            # encode text
            text_encode = [BOS]
            for t in text:
                text_encode.extend(self.vocab.EncodeAsIds(t) + [SEP])
            text_encode[-1] = EOS
            
            self.data[chat_id]['text_encode'] = text_encode
            self.data[chat_id]['text_words'] = [self.vocab.DecodeIds(tid) for tid in text_encode]
            self.data[chat_id]['is_augmented'] = False
    
    def augment_data(self, augment_topn, augment_threshold):
        # collect tagged data
        tagged_data = [TaggedDocument(words=self.data[chat_id]['text_words'], tags=[str(chat_id)]) for chat_id in self.data]
        
        # train doc2vec model
        model = Doc2Vec(vector_size=300, window=3, min_count=1, workers=4, epochs=100)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

        # find the last id
        data_augmented = {}
        id_cur = max(self.data.keys()) + 1

        for chat_id in self.data:
            # find neighbors with sim >= augment_threshold
            neighbor_ids = [int(neighbor_id) for neighbor_id, sim in model.docvecs.most_similar(str(chat_id), topn=augment_topn) if sim >= augment_threshold]

            # if there's speaker in the data, filter only the sample speaker
            if 'speaker_name' in self.data[chat_id]:
                neighbor_ids = [neighbor_id for neighbor_id in neighbor_ids if self.data[neighbor_id]['speaker_name'] == self.data[chat_id]['speaker_name']]

            # augment neighbor data
            for neighbor_id in neighbor_ids:
                neighbor = self.data[chat_id]
                neighbor['id'] = id_cur
                neighbor['text'] = self.data[neighbor_id]['text']
                neighbor['is_augmented'] = True

                data_augmented[id_cur] = neighbor
                id_cur += 1
        
        # merge
        print("before augment:", len(self.data))
        self.data = {**self.data, **data_augmented}
        print("after augment:", len(self.data))
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        answer_id = self.ids[index]
        answer = self.data[answer_id]

        question_id = answer['question_id']
        question = self.data[question_id]

        return {
            'data': (torch.tensor(question['text_encode']), torch.tensor(answer['text_encode'])),
            'is_augmented': answer['is_augmented']
        }


def collate_fn(inputs):
    x_enc = []
    x_dec = []

    for data in inputs:
        x_enc.append(data['data'][0])
        x_dec.append(data['data'][1])

    x_enc = torch.nn.utils.rnn.pad_sequence(x_enc, batch_first=True, padding_value=PAD)
    x_dec = torch.nn.utils.rnn.pad_sequence(x_dec, batch_first=True, padding_value=PAD)
    
    return [x_enc, x_dec]