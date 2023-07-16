import argparse
import torch
import sentencepiece as spm
import config
from constant import *
from model.classifier import Classifier


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', dest='inputs', nargs='+', required=True)
    args = parser.parse_args()

    # vocab
    vocab = spm.SentencePieceProcessor()
    vocab.load(PATH_VOCAB)

    # model
    model = Classifier()
    model = model.to(config.device)
    model.load_state_dict(torch.load(PATH_WEIGHT, map_location=config.device))

    # inputs
    x_enc = torch.tensor([BOS + vocab.EncodeAsIds(args.inputs[0]) + EOS]).to(config.device)
    x_dec = torch.tensor([BOS]).to(config.device)

    # forward
    model.eval()

    for _ in range(50):
        predict = model(x_enc, x_dec)
        last_word = torch.argmax(predict[:, -1:, :], dim=-1)

        if last_word.item() == EOS[0]:
            break

        x_dec = torch.cat([x_dec, last_word], dim=1)
    
    print(vocab.DecodeIds(x_enc.cpu().tolist()[0]))
    print(vocab.DecodeIds(x_dec.cpu().tolist()[0]))
