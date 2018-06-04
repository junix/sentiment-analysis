import os
import torch
import torch.nn.functional as F

from conf import run_device
from .net import Net

dump_path = os.path.dirname(os.path.abspath(__file__)) + '/model.pt'


def predict(net, text):
    net.eval()
    with torch.no_grad():
        hidden = net.init_hidden()
        score = net.forward(text, hidden)
        score = F.sigmoid(score).item()
        return 1 if score > 0.5 else 0


def load_predict():
    net = Net.load(dump_path)
    net.move_to_device(run_device())

    def _pred(text):
        return predict(net, text)

    return _pred
