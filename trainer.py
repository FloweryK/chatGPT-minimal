import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from constant import *
torch.set_printoptions(linewidth=10000)


def decodeSample(prefix, vocab, sample):
    print(prefix)
    print(vocab.DecodeIds(sample))
    print([vocab.DecodeIds([piece]) for piece in sample])
    print(sample, '\n')


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
                
                # sample
                sample_enc_input = x_enc[0][~x_enc[0].eq(0)]
                sample_dec_input = x_dec_input[0][~x_dec_input[0].eq(0)]
                sample_dec_target = x_dec_target[0][~x_dec_target[0].eq(0)]
                sample_predict = torch.argmax(predict, dim=1)[0][~x_dec_target[0].eq(0)]
                
                decodeSample("\nenc_input", self.vocab, sample_enc_input.tolist())
                decodeSample("dec_input", self.vocab, sample_dec_input.tolist())
                decodeSample("dec_target", self.vocab, sample_dec_target.tolist())
                decodeSample("dec_predict", self.vocab, sample_predict.tolist())
                
                # calculate performance
                predict_flatten = torch.argmax(predict, dim=1).flatten()
                target_flatten = x_dec_target.flatten()
                mask = ~target_flatten.eq(0)
                correct = torch.eq(predict_flatten[mask], target_flatten[mask]).detach().cpu().tolist()
                match.extend(correct)
                accuracy = np.mean(match) if match else 0
            
                # update progress bar
                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {losses[-1]:.3f} ({np.mean(losses):.3f}) | lr: {self.optimizer.lr:.8f} | Acc: {accuracy:.3f}")

            # tensorboard
            self.writer.add_scalar('Loss/train' if train else 'Loss/test', np.mean(losses), epoch)
            self.writer.add_scalar('Acc/train' if train else 'Acc/test', accuracy, epoch)

        # save model
        if train:
            torch.save(self.model.state_dict(), f'weights/model_{epoch}.pt')
        