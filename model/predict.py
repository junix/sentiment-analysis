import torch
import torch.nn.functional as F

from conf import run_device
from .train import dump_path
from .net import Net


def load_predict():
    net = Net.load(dump_path)
    net.eval()
    net.move_to_device(run_device())

    def predict(text):
        with torch.no_grad():
            hidden = net.init_hidden()
            score = net.forward(text, hidden)
            score = F.sigmoid(score).item()
            return 1 if score > 0.5 else 0

    return predict
