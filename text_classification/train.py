# coding: UTF-8
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch.nn.functional as F
from transformers.optimization import AdamW

from text_classification.infer import evaluate
from text_classification.data_utils import get_time_dif, Text_Classification_Dataset
import argparse
import logging
import math
import os
import random

from tokenization import Tokenizer4Bert

from absa.model import model_classes

import time
from time import strftime, localtime

import numpy
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(config, model, train_iter, dev_iter, test_iter):
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                model.train()
            total_batch += 1
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)

        self.train_set = Text_Classification_Dataset(opt.dataset_file['train'], opt, tokenizer)
        self.dev_set = Text_Classification_Dataset(opt.dataset_file['dev'], opt, tokenizer)
        self.test_set = Text_Classification_Dataset(opt.dataset_file['test'], opt, tokenizer)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            torch.nn.init.xavier_uniform_(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def train(self):
        # Loss and Optimizer
        begin_time = time.time()
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.lr)

        train_data_loader = DataLoader(dataset=self.train_set, batch_size=self.opt.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dataset=self.train_set, batch_size=self.opt.batch_size * 12, shuffle=False)
        test_data_loader = DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size * 12, shuffle=False)

        self._reset_params()

        best_f1 = 0
        global_step = 0
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            epoch_begin_time = time.time()
            # switch model to training mode
            for i_batch, (trains, labels) in enumerate(train_data_loader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                outputs = self.model(trains)
                self.model.zero_grad()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    ##todo
                    f1 = self.evaluate(test_data_loader)
                    if not os.path.exists(self.opt.model_dir):
                        os.mkdir(self.opt.model_dir)
                    if f1 > best_f1:
                        best_f1 = f1
                        torch.save(self.model.state_dict(), os.path.join(self.opt.model_dir, "model.pt"))
            logger.info('epoch{} cost time:: {:.4f}'.format(i_epoch, time.time() - epoch_begin_time))

        self.model.load_state_dict(torch.load(os.path.join(self.opt.model_dir, "model.pt")))
        test_f1 = self.evaluate(test_data_loader)
        logger.info('>> test_f1: {:.4f}'.format(test_f1))
        logger.info("all cost time: {:.4f}".format(time.time() - begin_time))

    def evaluate(self, data_loader):
        all_targets, all_output = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = t_batch['polarity'].to(self.opt.device)
                outputs = self.model(inputs)

                if all_targets is None:
                    all_targets = targets
                    all_output = outputs
                else:
                    all_targets = torch.cat((all_targets, targets), dim=0)
                    all_output = torch.cat((all_output, outputs), dim=0)
        y_true = all_targets.cpu()
        y_pred = torch.argmax(all_output, -1).cpu()
        f1 = metrics.f1_score(y_true, y_pred, average='weighted')
        report = metrics.classification_report(y_true, y_pred, digits=4)
        print(report)
        return f1


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert', type=str)
    parser.add_argument('--model_dir', default='ckpt', type=str)
    parser.add_argument('--dataset', default='THUCNews', type=str, help='camera')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--num_epoch', default=5, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='../pretrain_model/bert/', type=str)
    parser.add_argument('--max_seq_len', default=32, type=int)
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    dataset_files = {
        'THUCNews': {
            'train': 'data/THUCNews/train.txt',
            'test': 'data/THUCNews/test.txt',
            'dev': 'data/THUCNews/dev.txt'
        },
    }
    if opt.model_name not in model_classes:
        print("model_name does not support!")
        return
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    trainer = Trainer(opt)
    trainer.train()


if __name__ == '__main__':
    main()
