from collections import defaultdict, Counter
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from constant import *


def get_bleu(reference, candidate, N=4):
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
        

class Evaluator:
    def __init__(self, model, criterion, optimizer, vocab):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.vocab = vocab
        self.writer = SummaryWriter()
    
    def run_epoch(self, epoch, dataloader, device):
        metric = defaultdict(list)

        self.model.eval()
        
        with tqdm(total=len(dataloader), desc=f"'Eval' | epoch {epoch}") as pbar:
            for data in dataloader:
                # load input, label
                x_enc, x_dec = (x.to(device) for x in  data)
                x_dec_input = torch.tensor([[BOS]]).to(device)

                # predict
                for _ in range(50):
                    predict = self.model(x_enc, x_dec_input)
                    last_word = torch.argmax(predict[:, :, -1:], dim=1)

                    x_dec_input = torch.cat([x_dec_input, last_word], dim=1)
                    
                    if last_word.item() == EOS:
                        break

                # get bleu
                reference = x_dec[0, 1:-1].tolist()
                candidate = x_dec_input[0, 1:-1].tolist()
                metric['bleu'].append(get_bleu(reference, candidate))
                bleu = np.mean(metric['bleu']) if metric['bleu'] else 0

                # update progress bar
                pbar.update(1)
                pbar.set_postfix_str(f"BLEU: {bleu:.3f}")

            # tensorboard
            self.writer.add_scalar('BLEU/eval', bleu, epoch)
