import time
import torch.nn as nn
import torch.optim as optim
from yxt_nlp_toolkit.embedding.general import WordEmbedding

from dataset.imdb import *
from .net import Net
from .predict import predict

dump_path = os.path.join(os.path.dirname(__file__), 'model.pt')
_glove_path = os.path.expanduser('~/nlp/glove.6B.200d.txt')


def train_and_dump(from_model=None, min_count=5, lr=1e-4):
    if from_model:
        net = Net.load(from_model)
    else:
        lang = get_lang(min_count=min_count)
        net = Net(lang=lang)
        embedding = WordEmbedding(_glove_path)
        net.init_params(pre_trained_wv=embedding)

    net.move_to_device(run_device())
    do_train(net, lr)


def do_train(net, lr):
    optimizer = optim.SGD(net.params_without_embedding(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    train_dataset = load_dataset(net.lang, seg='train', move_to_run_device=True)
    test_dataset = load_dataset(net.lang, seg='test')
    print('load dataset ok')
    total_count, total_loss, started_at = 1, .0, int(time.time())
    for epoch in range(15):
        for text, label in train_dataset:
            net.train()
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
                current = int(time.time())
                print(total_count, '=>', total_loss, 'consumed:', current - started_at, 'second')
                total_loss, started_at = .0, current
            if total_count % 20000 == 0:
                print('accu=', accu(net, test_dataset))
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
