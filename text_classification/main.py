# coding: UTF-8
import argparse
import os
import sys
import time
from importlib import import_module

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import numpy as np
import torch

from text_classification.train import train
from text_classification.data_utils import get_time_dif, Text_Classification_Dataset

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default='bert', type=str, help='choose a model: Bert, ERNIE')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # bert
    x = import_module('model.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start_time = time.time()
    print("Loading data...")
    train_iter = Text_Classification_Dataset(config.train_path, config)
    dev_iter = Text_Classification_Dataset(config.dev_path, config)
    test_iter = Text_Classification_Dataset(config.test_path, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
