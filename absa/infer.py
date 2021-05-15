import argparse
import os
import sys

from tokenization import Tokenizer4Bert

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from absa.model import model_classes

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel

from absa.data_utils import ABSA_Test_Dataset, input_colses


class Infer:
    """A simple inference example"""

    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(bert, opt).to(opt.device)

        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()

    def predict(self, data):
        lines = []
        for item in data:
            lines.append(item[0])
            lines.append(item[1])
        test_dataset = ABSA_Test_Dataset(lines, self.tokenizer)
        test_data_loader = DataLoader(dataset=test_dataset, batch_size=self.opt.batch_size, shuffle=False)

        all_output = None
        with torch.no_grad():
            for i_batch, batch in enumerate(test_data_loader):
                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                if all_output is None:
                    all_output = outputs
                else:
                    all_output = torch.cat((all_output, outputs), dim=0)

        y_probs = F.softmax(all_output, dim=-1).cpu().numpy()
        y_pred = y_probs.argmax(axis=-1) - 1
        y_pred = y_pred.tolist()
        return y_pred


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lcf_bert', type=str)
    parser.add_argument('--model_dir', default='ckpt', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='../pretrain_model/bert/', type=str)
    parser.add_argument('--batch_size', default=768, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--local_context_focus', default='cdw', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=10, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()

    if opt.model_name not in model_classes:
        print("model_name does not support!")
        return

    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.state_dict_path = os.path.join(opt.model_dir, "model.pt")
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    infer = Infer(opt)

    data = [
        ['非常人性化的$T$啊', '设计'],
        ['使得$T$一般', '夜景拍摄能力'],
    ]
    pred = infer.predict(data)
    print(pred)


if __name__ == '__main__':
    main()
