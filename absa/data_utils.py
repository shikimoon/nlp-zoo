# -*- coding: utf-8 -*-
import math

import numpy as np
from torch.utils.data import Dataset

from tokenization import pad_and_truncate

input_colses = {
    'bert_spc': ['global_context_indices', 'global_segments_indices'],
    'aen_bert': ['local_context_indices', 'aspect_bert_indices'],
    'lcf_bert': ['global_context_indices', 'global_segments_indices', 'local_context_indices',
                 'aspect_bert_indices'],
}


def cut_text(text_left, aspect, text_right, max_len):
    all_len = len(text_left) + len(aspect) * 2 + len(text_right) + 3
    if all_len <= max_len:
        return text_left, text_right
    max_len_side = math.floor((max_len - len(aspect) * 2 - 3) / 2)
    if len(text_left) <= max_len_side:
        text_right = text_right[0:max_len - len(aspect) * 2 - 3 - len(text_left)]
    elif len(text_right) <= max_len_side:
        text_left = text_left[len(text_left) - (max_len - len(aspect) * 2 - 3 - len(text_right)):len(text_left)]
    else:
        text_left = text_left[len(text_left) - max_len_side:len(text_left)]
        text_right = text_right[0:max_len_side]
    return text_left, text_right


class ABSA_Train_Dataset(Dataset):
    def __init__(self, data_path, tokenizer):
        fin = open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            lines[i] = lines[i].replace(" ", "")
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            aspect = aspect.replace(" ", "")
            polarity = lines[i + 2].strip()
            text_left, text_right = cut_text(text_left, aspect, text_right, tokenizer.max_seq_len)

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)
            polarity = int(polarity) + 1

            text_len = np.sum(text_indices != 0)
            global_context_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            global_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            global_segments_indices = pad_and_truncate(global_segments_indices, tokenizer.max_seq_len)

            local_context_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'global_context_indices': global_context_indices,
                'global_segments_indices': global_segments_indices,
                'local_context_indices': local_context_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'polarity': polarity,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSA_Test_Dataset(Dataset):
    def __init__(self, lines, tokenizer):
        all_data = []
        for i in range(0, len(lines), 2):
            lines[i] = lines[i].replace(" ", "")
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            aspect = aspect.replace(" ", "")
            text_left, text_right = cut_text(text_left, aspect, text_right, tokenizer.max_seq_len)

            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            aspect_len = np.sum(aspect_indices != 0)

            text_len = np.sum(text_indices != 0)
            global_context_indices = tokenizer.text_to_sequence(
                '[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
            global_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
            global_segments_indices = pad_and_truncate(global_segments_indices, tokenizer.max_seq_len)

            local_context_indices = tokenizer.text_to_sequence(
                "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
            aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

            data = {
                'global_context_indices': global_context_indices,
                'global_segments_indices': global_segments_indices,
                'local_context_indices': local_context_indices,
                'aspect_bert_indices': aspect_bert_indices,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
