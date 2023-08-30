import os
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from constant import *
from pprint import pprint


class ChatDatasetBase(Dataset):
    def __init__(self, path_data, run_dir, n_vocab, augment, topn, threshold):
        super().__init__()

        # save run_dir
        self.run_dir = run_dir

        # load and encode data
        self.data = {}
        self.vocab = spm.SentencePieceProcessor()
        self.load_data(path_data)
        self.load_vocab(n_vocab)
        self.encode_data()
        
        # train doc2vec
        if augment:
            self.find_neighbors(topn, threshold)
            self.augment_data()

        # filter answers with no question
        self.ids = [value['id'] for value in self.data.values() if value['question_id']]
        self.len = len(self.ids)
    
    def load_data(self):
        raise NotImplementedError()
    
    def load_vocab(self, n_vocab):
        # set paths
        path_vocab = os.path.join(self.run_dir, 'vocab.model')
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

        # delete unnecessary files
        os.remove(path_txt)
        os.remove(path_prefix + '.vocab')

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

    def find_neighbors(self, topn, threshold):
        # collect tagged data
        tagged_data = []

        for chat_id in self.data:
            tagged_data.append(TaggedDocument(self.data[chat_id]['text_words'], [str(chat_id)]))
        
        # train doc2vec model
        model = Doc2Vec(vector_size=300, window=3, min_count=1, workers=4, epochs=100)
        model.build_vocab(tagged_data)
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

        # find neighbors
        for chat_id in self.data:
            neighbor_ids = [int(neighbor_id) for neighbor_id, sim in model.docvecs.most_similar(str(chat_id), topn=topn) if sim >= threshold]

            # if there's speaker in the data, filter only the sample speaker
            if 'speaker_name' in self.data[chat_id]:
                neighbor_ids = [neighbor_id for neighbor_id in neighbor_ids if self.data[neighbor_id]['speaker_name'] == self.data[chat_id]['speaker_name']]

            self.data[chat_id]['neighbor_ids'] = neighbor_ids
    
    def augment_data(self):
        # find the last id
        id_cur = max(self.data.keys()) + 1

        # make augmented dataset
        data_augmented = {}
        for chat_id in self.data:
            # get neighbor ids
            neighbor_ids = self.data[chat_id]['neighbor_ids']

            # augment neighbor data
            for neighbor_id in neighbor_ids:
                neighbor = self.data[chat_id]
                neighbor['id'] = id_cur
                neighbor['text'] = self.data[neighbor_id]['text']

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

        return (
            torch.tensor(question['text_encode']),
            torch.tensor(answer['text_encode']),
        )


def collate_fn(inputs):
    x_enc, x_dec = list(zip(*inputs))

    x_enc = torch.nn.utils.rnn.pad_sequence(x_enc, batch_first=True, padding_value=PAD)
    x_dec = torch.nn.utils.rnn.pad_sequence(x_dec, batch_first=True, padding_value=PAD)
    
    return [x_enc, x_dec]