import torch
import torch.nn as nn
from yxt_nlp_toolkit.utils import tokenizer

from conf import run_device


class Net(nn.Module):
    def __init__(self, lang, embedding_dim=200, hidden_size=512, rnn_type="gru"):
        super(Net, self).__init__()
        self.rnn_num_layers = 2
        self.rnn_bidirectional = True
        self.hidden_size = hidden_size
        self.lang = lang
        self.embedding = nn.Embedding(num_embeddings=lang.vocab_size, embedding_dim=embedding_dim)
        self.rnn_type = rnn_type
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=self.hidden_size,
                dropout=0.5,
                num_layers=self.rnn_num_layers,
                bidirectional=self.rnn_bidirectional)
        else:
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=self.hidden_size,
                dropout=0.5,
                num_layers=self.rnn_num_layers,
                bidirectional=self.rnn_bidirectional)
        self.decoder = nn.Linear(hidden_size * 2 if self.rnn_bidirectional else 1, out_features=1)

    def move_to_device(self, device):
        if device.type == 'cpu':
            self.cpu()
        else:
            self.cuda()

    def init_hidden(self):
        bidirect = 2 if self.rnn_bidirectional else 1
        if self.rnn_type == "lsmt":
            return (
                torch.zeros(self.rnn_num_layers * bidirect, 1, self.hidden_size, device=run_device()),
                torch.zeros(self.rnn_num_layers * bidirect, 1, self.hidden_size, device=run_device())
            )
        else:
            return torch.zeros(self.rnn_num_layers * bidirect, 1, self.hidden_size, device=run_device())

    def init_params(self, pre_trained_wv=None):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, .0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        if pre_trained_wv is not None:
            weight = self.embedding.weight.detach_().cpu().numpy()
            self.lang.build_embedding(wv=pre_trained_wv, out_embedding=weight)
            self.embedding.weight.data = torch.tensor(weight, dtype=torch.float)

    def forward(self, words, hidden):
        if isinstance(words, str):
            words = tokenizer(text=words, use_lib='naive')
            words = self.lang.to_indices(words)
            assert len(words) > 0, "len(words) should > 0"
            words = torch.tensor(words, dtype=torch.long, device=run_device())
        words = words.to(run_device()).view(-1)
        with torch.no_grad():
            input = self.embedding(words).view(len(words), 1, -1)
        output, hidden = self.rnn(input, hidden)
        output = output[-1].view(-1)
        score = self.decoder(output)
        return score

    def params_without_embedding(self):
        for name, para in self.named_parameters():
            if 'embedding' not in name:
                yield para

    def __repr__(self):
        return '<Net>'

    def dump(self, dump_path):
        with open(dump_path, 'wb') as f:
            torch.save(self, f)

    @classmethod
    def load(cls, dump_path):
        with open(dump_path, 'rb') as f:
            return torch.load(f, map_location=lambda storage, loc: storage)
