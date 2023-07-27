import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, random_split
import config
from constant import *
from dataset import KakaotalkDataset, collate_fn
from model.classifier import Classifier
from trainer import Trainer


class AdamWarmup:
    def __init__(self, optimizer, model_size, warmup_steps):
        self.optimizer = optimizer
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.lr = 0
        
    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))
        
    def step(self):
        # Increment the number of steps each time we call the step function
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # update the learning rate
        self.lr = lr
        self.optimizer.step()   


if __name__ == '__main__':
    # load vocab
    vocab = spm.SentencePieceProcessor()
    vocab.Load(PATH_VOCAB)

    # dataset
    dataset = KakaotalkDataset(vocab, PATH_DATA, target_speaker=config.target_speaker)
    train_size = int(config.rate_split * len(dataset))
    trainset, testset = random_split(dataset, [train_size, len(dataset) - train_size])

    # dataloader
    trainloader = DataLoader(trainset, batch_size=config.n_batch, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=config.n_batch, shuffle=True, collate_fn=collate_fn)

    # model
    model = Classifier(config)
    model = model.to(config.device)
    print("model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # trainer 
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=config.label_smoothing)
    adam = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    optimizer = AdamWarmup(adam, config.d_emb, config.warmup_steps)
    trainer = Trainer(model, criterion, optimizer, vocab)

    # train
    for epoch in range(config.n_epoch):
        trainer.run_epoch(epoch, trainloader, device=config.device, train=True)
        trainer.run_epoch(epoch, testloader, device=config.device, train=False)