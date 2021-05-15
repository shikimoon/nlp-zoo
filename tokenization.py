# -*- coding: utf-8 -*-

import numpy as np

from transformers import BertTokenizer


def pad_and_truncate(sequence, max_seq_len, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(max_seq_len) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-max_seq_len:]
    else:
        trunc = sequence[:max_seq_len]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
