import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from yxt_nlp_toolkit.embedding.general import WordEmbedding

from dataset.imdb import *
from conf import run_device
from .net import Net
from .predict import predict

dump_path = os.path.dirname(os.path.abspath(__file__)) + '/model.pt'
_glove_path = os.path.expanduser('~/nlp/glove.6B.200d.txt')


def train_and_dump(min_count=5):
    lang = get_lang(min_count=min_count)
    net = Net(lang=lang)
    embedding = WordEmbedding(_glove_path)
    weight = net.embedding.weight.detach_().cpu().numpy()
    lang.build_embedding(wv=embedding, out_embedding=weight)
    net.embedding.weight.data = torch.tensor(weight, dtype=torch.float)

    net.move_to_device(run_device())
    do_train(net)


def do_train(net):
    optimizer = optim.SGD(net.params_without_embedding(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train_dataset = DataLoader(list(read_imdb(seg='train')), batch_size=1, shuffle=True)
    test_dataset = DataLoader(list(read_imdb(seg='test')), batch_size=1, shuffle=False)
    total_count = 0
    total_loss = .0
    for epoch in range(5):
        for text, label in train_dataset:
            net.train()
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
                print(total_loss, 'accu=', accu(net, test_dataset))
                total_loss = .0
            if total_count % 5000 == 0:
                net.dump(dump_path)
    net.dump(dump_path)


def accu(net, test_dataset):
    total = float(len(test_dataset))
    right_cnt = 0
    for text, label in test_dataset:
        net.eval()
        result = predict(net, text[0])
        label = label.long().item()
        if result == label:
            right_cnt += 1
    return right_cnt / total
