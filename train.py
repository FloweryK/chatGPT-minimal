import os
import datetime
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from model.classifier import Classifier
from utils.constant import *
from utils.config import Config
from utils.dataset import ChatDataset
from utils.scheduler import WarmupScheduler
from utils.trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--path_data', required=True)
    parser.add_argument('-c', '--path_config', default='./config.json')
    return parser.parse_args()


def create_directories(config):
    base_dir = os.path.join(
        'runs',
        f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}' \
        + f'_vocab={config.n_vocab}' + f'_batch={config.n_batch}' \
        + f'_accum={config.n_accum}' \
        + f'_amp={config.is_amp}' \
        + f'_warmup={config.warmup_steps}' \
        + f'_demb={config.d_emb}'
    )
    os.makedirs('runs', exist_ok=True)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # parse arguments
    args = parse_arguments()
    path_data = args.path_data
    path_config = args.path_config

    config = Config(path_config)
    base_dir = create_directories(config)
    path_prefix = os.path.join(base_dir, 'vocab')
    path_weight = os.path.join(base_dir, 'model.pt')
        
    # dataset
    dataset = ChatDataset(
        path_data=path_data,
        path_prefix=path_prefix,
        n_vocab=config.n_vocab,
    )
    train_size = int(config.r_split * len(dataset))
    trainset, testset = random_split(dataset, [train_size, len(dataset) - train_size])

    # dataloader
    trainloader = DataLoader(trainset, batch_size=config.n_batch, shuffle=True, collate_fn=dataset.collate_fn)
    testloader = DataLoader(testset, batch_size=config.n_batch, shuffle=True, collate_fn=dataset.collate_fn)

    # model
    model = Classifier(config)
    model = model.to(config.device)

    # criterion, optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, warmup_steps=config.warmup_steps, d_model=config.d_emb)

    # scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.is_amp)

    # writer
    writer = SummaryWriter(log_dir=base_dir)

    # trainer
    trainer = Trainer(model, criterion, scaler, optimizer, scheduler, writer, dataset.tokenizer)

    # train
    for epoch in range(config.n_epoch):
        trainer.run_epoch(epoch, trainloader, config.device, True, config.is_amp, config.n_accum)
        trainer.run_epoch(epoch, testloader, config.device, False, config.is_amp, config.n_accum)
        trainer.save_weight(path_weight)