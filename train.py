import argparse
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import config
from constant import *
from dataset.base import collate_fn
from dataset.movie_corpus import MovieCorpusDataset
from model.classifier import Classifier
from trainer import Trainer


def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, d_model):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.current_step = 0
        super().__init__(optimizer)
    
    def get_lr(self):
        self.current_step += 1
        lr = self.d_model ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))

        return [lr for _ in self.base_lrs]


if __name__ == '__main__':
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-v', '--vocab', required=False)
    args = parser.parse_args()
    path_data = args.data
    path_vocab = args.vocab

    # dataset
    dataset = MovieCorpusDataset(config.n_vocab, path_data, path_vocab)
    train_size = int(config.r_split * len(dataset))
    trainset, testset = random_split(dataset, [train_size, len(dataset) - train_size])

    # dataloader
    trainloader = DataLoader(trainset, batch_size=config.n_batch, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=config.n_batch, shuffle=True, collate_fn=collate_fn)

    # model
    model = Classifier(config)
    model = model.to(config.device)
    print("model parameters:", get_model_parameters(model))

    # criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, warmup_steps=config.warmup_steps, d_model=config.d_emb)

    # writer
    writer = SummaryWriter()

    # trainer
    trainer = Trainer(model, criterion, optimizer, scheduler, writer)

    # train
    for epoch in range(config.n_epoch):
        trainer.run_epoch(epoch, trainloader, device=config.device, train=True, use_autocast=config.use_autocast, n_accum=config.n_accum)
        trainer.run_epoch(epoch, testloader, device=config.device, train=False, use_autocast=config.use_autocast, n_accum=config.n_accum)