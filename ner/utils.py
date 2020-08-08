# -*- encoding: utf-8 -*-

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

bert_model = 'bert'
tokenizer = BertTokenizer.from_pretrained(bert_model)
VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-INF', 'I-INF', 'B-PAT', 'I-PAT', 'B-OPS',
         'I-OPS', 'B-DSE', 'I-DSE', 'B-DRG', 'I-DRG', 'B-LAB', 'I-LAB')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 256 - 2


class NerDataset(Dataset):
    def __init__(self, f_path):
        with open(f_path, 'r', encoding='utf-8') as fr:
            entries = fr.read().strip().split('\n\n')
        sents, tags_li = [], []  # list of lists
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            if len(words) > MAX_LEN:
                # 先对句号分段
                word, tag = [], []
                for char, t in zip(words, tags):

                    if char != '。':
                        if char != '\ue236':  # 测试集中有这个字符
                            word.append(char)
                            tag.append(t)
                    else:
                        sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                        tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                        word, tag = [], []
                        # 最后的末尾
                if len(word):
                    sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                    tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                    word, tag = [], []
            else:
                sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
                tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
        self.sents, self.tags_li = sents, tags_li

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx]
        x, y = [], []
        is_heads = []
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0] * (len(tokens) - 1)
            t = [t] + ['<PAD>'] * (len(tokens) - 1)
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)
        assert len(x) == len(y) == len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen

    def __len__(self):
        return len(self.sents)


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)

    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens


def merge_maps(dict1, dict2):
    """用于合并两个word2id或者两个tag2id"""
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = len(dict1)
    return dict1


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model


# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists


def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list


def word2features(sent, i):
    """抽取单个字的特征"""
    word = sent[i]
    prev_word = "<s>" if i == 0 else sent[i - 1]
    next_word = "</s>" if i == (len(sent) - 1) else sent[i + 1]
    # 使用的特征：
    # 前一个词，当前词，后一个词，
    # 前一个词+当前词， 当前词+后一个词
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word + word,
        'w:w+1': word + next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    """抽取序列特征"""
    return [word2features(sent, i) for i in range(len(sent))]


# ******** LSTM模型 工具函数*************

def tensorized(batch, maps):
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths


def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)),
                     key=lambda k: len(pairs[k][0]),
                     reverse=True)
    pairs = [pairs[i] for i in indices]
    # pairs.sort(key=lambda pair: len(pair[0]), reverse=True)

    word_lists, tag_lists = list(zip(*pairs))

    return word_lists, tag_lists, indices


def cal_loss(logits, targets, tag2id):
    """计算损失
    参数:
        logits: [B, L, out_size]
        targets: [B, L]
        lengths: [B]
    """
    PAD = tag2id.get('<pad>')
    assert PAD is not None

    mask = (targets != PAD)  # [B, L]
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)

    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)

    return loss


# FOR BiLSTM-CRF


def cal_lstm_crf_loss(crf_scores, targets, tag2id):
    """计算双向LSTM-CRF模型的损失
    该损失函数的计算可以参考:https://arxiv.org/pdf/1603.01360.pdf
    """
    pad_id = tag2id.get('<pad>')
    start_id = tag2id.get('<start>')
    end_id = tag2id.get('<end>')

    device = crf_scores.device

    # targets:[B, L] crf_scores:[B, L, T, T]
    batch_size, max_len = targets.size()
    target_size = len(tag2id)

    # mask = 1 - ((targets == pad_id) + (targets == end_id))  # [B, L]
    mask = (targets != pad_id)
    lengths = mask.sum(dim=1)
    targets = indexed(targets, target_size, start_id)

    # # 计算Golden scores方法１
    # import pdb
    # pdb.set_trace()
    targets = targets.masked_select(mask)  # [real_L]

    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size * target_size).contiguous()

    golden_scores = flatten_scores.gather(
        dim=1, index=targets.unsqueeze(1)).sum()

    # 计算golden_scores方法２：利用pack_padded_sequence函数
    # targets[targets == end_id] = pad_id
    # scores_at_targets = torch.gather(
    #     crf_scores.view(batch_size, max_len, -1), 2, targets.unsqueeze(2)).squeeze(2)
    # scores_at_targets, _ = pack_padded_sequence(
    #     scores_at_targets, lengths-1, batch_first=True
    # )
    # golden_scores = scores_at_targets.sum()

    # 计算all path scores
    # scores_upto_t[i, j]表示第i个句子的第t个词被标注为j标记的所有t时刻事前的所有子路径的分数之和
    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        # 当前时刻 有效的batch_size（因为有些序列比较短)
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t,
                                           t, start_id, :]
        else:
            # We add scores at current timestep to scores accumulated up to previous
            # timestep, and log-sum-exp Remember, the cur_tag of the previous
            # timestep is the prev_tag of this timestep
            # So, broadcast prev. timestep's cur_tag scores
            # along cur. timestep's cur_tag dimension
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] +
                scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1
            )
    all_path_scores = scores_upto_t[:, end_id].sum()

    # 训练大约两个epoch loss变成负数，从数学的角度上来说，loss = -logP
    loss = (all_path_scores - golden_scores) / batch_size
    return loss


def indexed(targets, tagset_size, start_id):
    """将targets中的数转化为在[T*T]大小序列中的索引,T是标注的种类"""
    batch_size, max_len = targets.size()
    for col in range(max_len - 1, 0, -1):
        targets[:, col] += (targets[:, col - 1] * tagset_size)
    targets[:, 0] += (start_id * tagset_size)
    return targets


def test():
    dir = "/Users/mlamp/Desktop/文档/data/Mengniu_batch3/"
    # dir = "/Users/mlamp/Desktop/文档/data/蒙牛关系抽取/Mengniu_batch3/"
    output_file = "/Users/mlamp/Desktop/1.txt"
    fp = open(output_file, "w", encoding="utf-8")
    write_lines = []
    list_file = os.listdir(dir)
    for i in range(len(list_file)):
        file_name = list_file[i]
        flags = file_name.split(".")
        if flags[1] == "txt":
            match_file_name = flags[0] + ".ann"
            tmp_fp1 = open(dir + file_name, "r", encoding="utf-8")
            tmp_fp2 = open(dir + match_file_name, "r", encoding="utf-8")
            lines1 = tmp_fp1.readlines()
            lines2 = tmp_fp2.readlines()
            new_line = lines1[0].strip()
            entities = []
            for line in lines2:
                if line.startswith("T"):
                    flags = line.strip().split("\t")
                    flags1 = flags[1].split(" ")
                    entity = [flags1[1], flags1[2], flags[2], flags1[0]]
                    entities.append(entity)
            entities.sort(key=lambda x: x[0])
            for entity in entities:
                new_line += ("\t" + entity[2] + ":::" + entity[3])
            write_lines.append(new_line + "\n")
            tmp_fp1.close()
            tmp_fp2.close()
    fp.writelines(write_lines)
    fp.close()


def test1():
    file = "/Users/mlamp/Desktop/1.txt"
    fp = open(file, "r", encoding="utf-8")
    dir = "/Users/mlamp/Desktop/文档/data/data/"
    all_data = {}
    for line in fp.readlines():
        flags = line.strip().split("\t")
        if len(flags) == 1:
            continue
        for entity in flags[2:]:
            flags1 = entity.strip().split(":::")
        if flags1[1] not in all_data:
            all_data[flags1[1]] = {}
        if flags1[0] not in all_data[flags1[1]]:
            all_data[flags1[1]][flags1[0]] = 1
        else:
            all_data[flags1[1]][flags1[0]] += 1
    for key, value in all_data.items():
        tmp_file = open(dir + key + ".txt", "w", encoding="utf-8")
        write_lines = []
        value = sorted(value.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        for entity_value, count in value:
            write_lines.append(entity_value + "\t" + str(count) + "\n")
        tmp_file.writelines(write_lines)
    fp.close()

if __name__ == "__main__":
    # train_file = "data/train.txt"
    # train_dataset = NerDataset(train_file)

    test1()
