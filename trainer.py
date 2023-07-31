import time
from collections import Counter
import numpy as np
import torch
from tqdm import tqdm
from constant import *


def get_match(x_dec_target, predict):
    target_flatten = x_dec_target.flatten()
    predict_flatten = predict.flatten()
    mask = ~target_flatten.eq(PAD)

    correct = torch.eq(predict_flatten[mask], target_flatten[mask]).detach().cpu().tolist()
    return correct


def get_bleu(reference, candidate, N=4):
    mask = ~reference.eq(PAD)
    reference = reference[mask].tolist()
    candidate = candidate[mask].tolist()

    ps = []

    for n in range(1, min(N+1, len(candidate) + 1)):
        ngram_r = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
        ngram_c = Counter([tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1)])

        # modified precision
        p = sum((ngram_r & ngram_c).values()) / sum(ngram_c.values())

        # weight
        w = 1 / N

        ps.append(p ** w)
    
    # brevity penalty
    bp = min(1, np.exp(1- len(reference) / (len(candidate) + 1e-10)))

    bleu = bp * np.prod(ps)
    bleu *= 100

    return bleu


class Trainer:
    def __init__(self, model, criterion, optimizer, vocab, writer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.vocab = vocab
        self.writer = writer
    
    def run_epoch(self, epoch, dataloader, device, train=True, n_accum=1):
        losses = []
        match = []
        times = []
        bleus = []

        if train:
            self.model.train()
        else:
            self.model.eval()

        with tqdm(total=len(dataloader), desc=f"{'train' if train else 'test '} {epoch}") as pbar:
            for i, data in enumerate(dataloader):
                # perf counter: start
                t_start = time.perf_counter()

                # load input, label
                x_enc, x_dec = (x.to(device) for x in  data)
                x_dec_input = x_dec[:, :-1]
                x_dec_target = x_dec[:, 1:]

                # autocast
                predict = self.model(x_enc, x_dec_input)

                # loss
                loss = self.criterion(predict, x_dec_target)
                losses.append(loss.item())
                loss = loss / n_accum

                # update model
                if train:
                    loss.backward()

                    # gradient accumulation
                    if ((i+1) % n_accum == 0) or (i+1 == len(dataloader)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                # perf counter: end
                t_end = time.perf_counter()
                times.append(t_end - t_start)

                # extract logits from predict
                predict = torch.argmax(predict, dim=1)
                
                # get accuracy
                match += get_match(x_dec_target, predict)
                accuracy = np.mean(match)

                # get bleu
                bleus += [get_bleu(ref, cand) for ref, cand in zip(x_dec_target, predict)]
                bleu = np.mean(bleus)

                # get memory
                mem_info = torch.cuda.mem_get_info()  if torch.cuda.is_available() else [0, 0]
                memory = (mem_info[1] - mem_info[0]) / 1024**3
            
                # update progress bar
                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {losses[-1]:.2f} ({np.mean(losses):.2f}) | lr: {self.optimizer.lr:.6f} | Acc: {accuracy:.3f} | bleu: {bleu:.1f} | {memory:.2f}GB | {np.mean(times) * 1000:.0f}ms")

            # save model
            if train and ((epoch + 1) % 5 == 0):
                torch.save(self.model.state_dict(), f'weights/model_{epoch}.pt')
            
            # tensorboard
            self.writer.add_scalar(f'Train/Loss' if train else 'Test/Loss', np.mean(losses), epoch)
            self.writer.add_scalar(f'Train/bleu' if train else 'Test/bleu', bleu, epoch)
            self.writer.add_scalar(f'Train/Acc' if train else 'Test/Acc', accuracy, epoch)
            self.writer.add_scalar(f'Train/memory' if train else 'Test/memory', memory, epoch)
            self.writer.add_scalar(f'Train/time_iter' if train else 'Test/time_iter', np.mean(times) * 1000, epoch)
            self.writer.add_scalar(f'Train/time_epoch' if train else 'Test/time_epoch', np.sum(times), epoch)

