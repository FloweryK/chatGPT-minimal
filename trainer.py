import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from constant import *


def get_match(x_dec_target, predict):
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
    
    def run_epoch(self, epoch, dataloader, device, train=True, n_accum=1):
        losses = []
        match = []

        if train:
            self.model.train()
        else:
            self.model.eval()

        with tqdm(total=len(dataloader), desc=f"{'train' if train else 'test '} {epoch}") as pbar:
            for i, data in enumerate(dataloader):
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

                    # gradient accumulation
                    if ((i+1) % n_accum == 0) or (i+1 == len(dataloader)):
                        self.optimizer.step()    

                # extract logits from predict
                predict = torch.argmax(predict, dim=1)
                
                # get accuracy
                match += get_match(x_dec_target, predict)
                accuracy = np.mean(match) if match else 0

                # get memory
                memory = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
            
                # update progress bar
                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {losses[-1]:.3f} ({np.mean(losses):.3f}) | lr: {self.optimizer.lr:.8f} | Acc: {accuracy:.3f} | {memory:.2f}GB")

            # save model
            if train and ((epoch + 1) % 10 == 0):
                torch.save(self.model.state_dict(), f'weights/model_{epoch}.pt')
            
            # tensorboard
            self.writer.add_scalar(f'Loss/train' if train else 'Loss/test', np.mean(losses), epoch)
            self.writer.add_scalar(f'Acc/train' if train else 'Acc/test', accuracy, epoch)

