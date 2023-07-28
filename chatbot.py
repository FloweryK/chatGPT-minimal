import os
import argparse
import torch
import sentencepiece as spm
import config
from constant import *
from model.classifier import Classifier


class Chatbot:
    def __init__(self, config):
        # config
        self.config = config

        # vocab
        self.vocab = spm.SentencePieceProcessor()
        self.vocab.load(config.path_vocab)

        # model
        self.model = Classifier(config)
        self.model = self.model.to(config.device)
        self.model.load_state_dict(torch.load(os.path.join(config.path_weight, f'model_{args.epoch}.pt'), map_location=config.device))
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
    parser.add_argument('-t', '--text', required=True)
    parser.add_argument('-e', '--epoch', required=True)
    args = parser.parse_args()

    # chatbot
    chatbot = Chatbot(config)

    # chat
    question, question_decode, answer, answer_decode = chatbot.chat(args.text)
    print(question)
    print(question_decode)
    print(answer)
    print(answer_decode)
