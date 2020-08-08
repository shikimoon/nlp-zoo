# -*- encoding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils import data

from ner.model.bert_crf import Bert_CRF
from ner.utils import NerDataset, pad, tag2idx, idx2tag

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Bert_CRF(tag2idx).cuda()
    print('Initial model Done')

    train_dataset = NerDataset(hp.train_file)
    eval_dataset = NerDataset(hp.valid_file)
    print('Load Data Done')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    print('Start Train...,')
    f1 = 0
    best_f1 = 0
    for epoch in range(1, hp.n_epochs + 1):  # 每个epoch对dev集进行测试

        diff_lr = hp.lr
        if f1 <= 0.7:
            diff_lr = diff_lr * 100
        elif 0.7 < f1 <= 0.8:
            diff_lr = diff_lr * 10
        else:
            diff_lr = diff_lr
        optimizer = optim.Adam(model.parameters(), lr=diff_lr)
        train_per_epoch(model, train_iter, optimizer, device)

        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(hp.model_dir): os.makedirs(hp.model_dir)
        precision, recall, f1 = eval(model, eval_iter, device)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(hp.model_dir, "model.pt"))


def train_per_epoch(model, iterator, optimizer, device):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        x = x.to(device)
        y = y.to(device)
        _y = y
        optimizer.zero_grad()
        loss = model.loss(x, y)  # logits: (N, T, VOCAB), y: (N, T)

        loss.backward()
        optimizer.step()

        if i == 0:
            print("=====sanity check======")
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")

        if i % 10 == 0:
            print(f"step: {i}, loss: {loss.item()}")


def eval(model, iterator, device):
    model.eval()
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            x = x.to(device)
            # y = y.to(device)

            _, y_hat = model(x)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("temp", 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true = np.array(
        [tag2idx[line.split()[1]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])
    y_pred = np.array(
        [tag2idx[line.split()[2]] for line in open("temp", 'r', encoding='utf-8').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred > 1])
    num_correct = (np.logical_and(y_true == y_pred, y_true > 1)).astype(np.int).sum()
    num_gold = len(y_true[y_true > 1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        if precision * recall == 0:
            f1 = 1.0
        else:
            f1 = 0

    os.remove("temp")

    print("precision=%.2f" % precision)
    print("recall=%.2f" % recall)
    print("f1=%.2f" % f1)
    return precision, recall, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--train_file", type=str, default="data/train.txt")
    parser.add_argument("--valid_file", type=str, default="data/dev.txt")
    parser.add_argument("--model_dir", type=str, default="data/model")
    hp = parser.parse_args()

    train()
