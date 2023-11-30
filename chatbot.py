import argparse
import torch
import sentencepiece as spm
from utils.constant import *
from utils.config import Config
from model.classifier_builtin import Classifier


class Chatbot:
    def __init__(self, config, path_vocab, path_weight):
        # config
        self.config = config

        # vocab
        self.vocab = spm.SentencePieceProcessor()
        self.vocab.load(path_vocab)

        # model
        self.model = Classifier(config)
        self.model = self.model.to(config.device)
        self.model.load_state_dict(torch.load(path_weight, map_location=config.device))
        self.model.eval()
    
    def chat(self, text, n_max=50):
        question = torch.tensor([[BOS] + self.vocab.EncodeAsIds(text) + [EOS]]).to(self.config.device)
        answer = torch.tensor([[BOS]]).to(self.config.device)

        for _ in range(n_max):
            predict = self.model(question, answer)
            last_word = torch.argmax(predict[:, :, -1:], dim=1)

            answer = torch.cat([answer, last_word], dim=1)
            
            if last_word.item() == EOS:
                break
        
        question = question.cpu().tolist()[0]
        answer = answer.cpu().tolist()[0]

        question_decode = self.vocab.DecodeIds(question)
        answer_decode = self.vocab.DecodeIds(answer)

        return question, question_decode, answer, answer_decode


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', required=True)
    parser.add_argument('-v', '--vocab', required=True)
    parser.add_argument('-c', '--path_config', default='./config.json')
    args = parser.parse_args()
    
    path_vocab = args.vocab
    path_weight = args.weight
    path_config = args.path_config

    # chatbot
    config = Config(path_config)
    chatbot = Chatbot(config, path_vocab, path_weight)

    # chat
    while True:
        text = input('text: ')
        question, question_decode, answer, answer_decode = chatbot.chat(text)
        print(question)
        print(question_decode)
        print(answer)
        print(answer_decode)
