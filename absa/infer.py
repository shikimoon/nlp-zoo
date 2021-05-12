import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel

from absa.data_utils import Tokenizer4Bert, ABSA_Train_Dataset, ABSA_Test_Dataset
from absa.model.aen import AEN_BERT
from absa.model.bert_spc import BERT_SPC
from absa.model.lcf_bert import LCF_BERT


class Predicter:
    """A simple inference example"""

    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            print("model_name does not support!")
            return
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()

    def predict(self, file_name):
        test_dataset = ABSA_Train_Dataset(file_name, self.tokenizer)
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
        y_pred = torch.argmax(all_output, -1).cpu().tolist()

        predict_file = "data/camera/infer.txt"
        predict_fp = open(predict_file, 'w', encoding='utf-8', newline='\n', errors='ignore')
        write_lines = []
        for result in y_pred:
            write_lines.append(str(result - 1) + "\n")
        predict_fp.writelines(write_lines)
        predict_fp.close()
        return y_pred

    def predict_example(self, data):
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
        print(y_probs)
        print(y_pred)
        return y_pred


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lcf_bert', type=str)
    parser.add_argument('--model_dir', default='ckpt', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='pretrain_model/bert/', type=str)
    parser.add_argument('--batch_size', default=768, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--local_context_focus', default='cdw', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=10, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()

    model_classes = {
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
    }
    input_colses = {
        'bert_spc': ['global_context_indices', 'global_segments_indices'],
        'aen_bert': ['local_context_indices', 'aspect_bert_indices'],
        'lcf_bert': ['global_context_indices', 'global_segments_indices', 'local_context_indices',
                     'aspect_bert_indices'],
    }

    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.state_dict_path = os.path.join(opt.model_dir, "model.pt")
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predicter = Predicter(opt)

    test_file = "absa/data/camera/test.txt"

    data = [
        [
            '如果是我我上H6，这个价位看销量靠口碑就可以了，不用在乎豪华性。居家车代步，稳定是首选。h6销量榜首纸的不必多说了，口碑看懂车帝评分，三者差不多，h6评分4.05,长安是4.03，荣威4.04，差别不大。但销量那么高评分还最好为啥不选H6?懂车帝视频拆车时长安$T$，相比之下哈弗良心的多，安全性高啊。荣威是干式双离合，用不长久。说到油耗，确实哈弗更费油，但如果家用的话一年一万公里也差不太多钱。',
            '偷工减料'],
        [
            '在选择蓝鸟之前也多车比较过， 丰田 卡罗拉( 参数| 询价)， 马自达 昂克赛拉( 参数| 询价)， 本田 思域( 参数| 询价)， 日产 轩逸( 参数| 询价)！最终选择18款蓝鸟智酷白。是因为别，就是这车$T$吸引我。悬浮式车顶路上少见。在路上看到别人开过以后就一见钟情。蓝鸟总来说性价比最高还是智酷版，对这个版本最满意就是没有配备胎压监测，最后自己去外面装一个胎压监测，1.6CVT变速箱算是日产标配吧，起步平顺，就是动力比较肉。都说这车省油，是我开对，还是怎么，我油耗就是高。座椅买车时候就包真皮，座椅颜色是红色黑色搭配，为就是我车中控对应。好，今天就说到这吧，',
            '外观'],
    ]
    pred = predicter.predict(test_file)
    # pred = predicter.predict_example(data)
