import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from constant import *


# TMP
def show_sample(vocab, x_enc, x_dec_input, x_dec_target, predict):
    def decode(name, sample, mask):
        sample = sample[~mask].tolist()

        print('\n' + name)
        print(vocab.DecodeIds(sample))
        print([vocab.DecodeIds([piece]) for piece in sample])
        print(sample)

    # extract logits from predict
    predict = torch.argmax(predict, dim=1)

    decode('x_enc', x_enc[0], x_enc[0].eq(PAD))
    decode('x_dec_input', x_dec_input[0], x_dec_input[0].eq(PAD))
    decode('x_dec_target', x_dec_target[0], x_dec_target[0].eq(PAD))
    decode('predict', predict[0], x_dec_target[0].eq(PAD))


def get_match(x_dec_target, predict):
    # extract logits from predict
    predict = torch.argmax(predict, dim=1)

    target_flatten = x_dec_target.flatten()
    predict_flatten = predict.flatten()
    mask = ~target_flatten.eq(PAD)

    correct = torch.eq(predict_flatten[mask], target_flatten[mask]).detach().cpu().tolist()
    return correct


class Trainer:
    def __init__(self, model, criterion, optimizer, vocab):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.vocab = vocab
        self.writer = SummaryWriter()
    
    def run_epoch(self, epoch, dataloader, device, train=True):
        losses = []
        match = []

        if train:
            self.model.train()
        else:
            self.model.eval()

        with tqdm(total=len(dataloader), desc=f"{'Train' if train else 'Test '} {epoch}") as pbar:
            for data in dataloader:
                # load input, label
                x_enc, x_dec = (x.to(device) for x in  data)
                x_dec_input = x_dec[:, :-1]
                x_dec_target = x_dec[:, 1:]

                # predict
                if train:
                    self.optimizer.optimizer.zero_grad()
                predict = self.model(x_enc, x_dec_input)

                # loss
                loss = self.criterion(predict, x_dec_target)
                losses.append(loss.item())

                # update model
                if train:
                    loss.backward()
                    self.optimizer.step()    

                # show samples
                show_sample(self.vocab, x_enc, x_dec_input, x_dec_target, predict)
                
                # get accuracy
                match += get_match(x_dec_target, predict)
                accuracy = np.mean(match) if match else 0
            
                # update progress bar
                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {losses[-1]:.3f} ({np.mean(losses):.3f}) | lr: {self.optimizer.lr:.8f} | Acc: {accuracy:.3f}")

            # save model
            if train and ((epoch + 1) % 10 == 0):
                torch.save(self.model.state_dict(), f'weights/model_{epoch}.pt')
            
            # tensorboard
            self.writer.add_scalar('Loss/train' if train else 'Loss/test', np.mean(losses), epoch)
            self.writer.add_scalar('Acc/train' if train else 'Acc/test', accuracy, epoch)

