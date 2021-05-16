# -*- coding: utf-8 -*-

import argparse
import logging
import math
import os
import random
import sys

from tokenization import Tokenizer4Bert

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from absa.model import model_classes

import time
from time import strftime, localtime

import numpy
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertModel

from absa.data_utils import ABSA_Train_Dataset, input_colses

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Trainer:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)

        self.train_set = ABSA_Train_Dataset(opt.dataset_file['train'], tokenizer)
        self.test_set = ABSA_Train_Dataset(opt.dataset_file['test'], tokenizer)

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
        optimizer = torch.optim.Adam(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.train_set, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.test_set, batch_size=self.opt.batch_size * 12, shuffle=False)

        self._reset_params()

        best_f1 = 0
        global_step = 0
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            epoch_begin_time = time.time()
            # switch model to training mode
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
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
    parser.add_argument('--model_name', default='lcf_bert', type=str)
    parser.add_argument('--model_dir', default='ckpt', type=str)
    parser.add_argument('--dataset', default='camera', type=str, help='camera')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=5, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='../pretrain_model/bert/', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--local_context_focus', default='cdw', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=10, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
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
        'camera': {
            'train': 'data/camera/train.txt',
            'test': 'data/camera/test.txt'
        },
    }
    if opt.model_name not in model_classes:
        print("model_name does not support!")
        return
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    trainer = Trainer(opt)
    trainer.train()


if __name__ == '__main__':
    main()
