import os

import torch
from torch.utils.data.dataloader import DataLoader

from yxt_nlp_toolkit.common import Vocab, Lang
from yxt_nlp_toolkit.utils import token_stream, tokenizer

_dataset_dir = os.path.expanduser('~/nlp/dataset/aclImdb')

POS_LABEL, NEG_LABEL = 1, 0


def _imdb_files(seg='train'):
    labels = 'pos', 'neg'
    for label in labels:
        label_dir = _dataset_dir + '/' + seg + '/' + label + '/'
        for file in os.listdir(label_dir):
            file_path = label_dir + '/' + file
            file_label = POS_LABEL if label == 'pos' else NEG_LABEL
            yield file_path, file_label


def read_imdb(seg='train'):
    for file, label in _imdb_files(seg):
        with open(file, 'r', encoding='utf8') as rf:
            review = rf.read().replace('\n', '').upper()
            yield review, label


def text_to_indices(text, lang):
    words = tokenizer(text=text, use_lib='naive')
    return lang.to_indices(words)


def load_dataset(lang, seg='train'):
    return DataLoader([(torch.tensor(text_to_indices(text, lang)), label)
                       for text, label in read_imdb(seg=seg)], batch_size=1, shuffle=True)


def read_vocab(min_count=1):
    files = tuple(file for file, _ in _imdb_files(seg='train'))
    tokens = token_stream(files, skip_space=True, use_lib='naive')
    vocab = Vocab(words=tokens)
    return vocab.shrink(min_count=min_count)


def get_lang(min_count=1):
    return Lang(words=read_vocab(min_count))
