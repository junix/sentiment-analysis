import torch


def run_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
