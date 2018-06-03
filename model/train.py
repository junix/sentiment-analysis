import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from dataset.imdb import *
from conf import run_device
from .net import Net

dump_path = os.path.dirname(os.path.abspath(__file__)) + '/model.pt'


def train_and_dump(min_count=5):
    lang = get_lang(min_count=min_count)
    net = Net(lang=lang)
    net.train()
    net.move_to_device(run_device())
    do_train(net)


def do_train(net):
    optimizer = optim.SGD(net.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_dataset = DataLoader(list(read_imdb(seg='train')), batch_size=1, shuffle=True)
    test_dataset = DataLoader(list(read_imdb(seg='test')), batch_size=1, shuffle=False)
    total_count = 0
    total_loss = .0
    for epoch in range(5):
        for text, label in train_dataset:
            text = text[0]
            net.zero_grad()
            hidden = net.init_hidden()
            score = net(text, hidden)
            label = label.view(score.shape).to(run_device(), dtype=torch.float)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach_().item()
            total_count += 1
            if total_count % 1000 == 0:
                print(total_loss)
                total_loss = .0
            if total_count % 5000 == 0:
                net.dump(dump_path)
    net.dump(dump_path)
