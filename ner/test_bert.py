# -*- encoding: utf-8 -*-

import os

import torch
from pytorch_pretrained_bert import BertTokenizer

from ner.model.bert_crf import Bert_CRF
from ner.utils import tag2idx, idx2tag

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CRF_MODEL_PATH = "data/model/model.pt"

class CRF(object):
    def __init__(self, crf_model, device='cpu'):
        self.device = torch.device(device)
        self.model = Bert_CRF(tag2idx)
        self.model.load_state_dict(torch.load(crf_model))
        self.model.to(device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained("bert")

    def predict(self, text):
        """Using CRF to predict label
        
        Arguments:
            text {str} -- [description]
        """
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text) + ['[SEP]']
        xx = self.tokenizer.convert_tokens_to_ids(tokens)
        xx = torch.tensor(xx).unsqueeze(0).to(self.device)
        _, y_hat = self.model(xx)
        pred_tags = []
        for tag in y_hat.squeeze():
            pred_tags.append(idx2tag[tag.item()])
        return pred_tags, tokens

    def parse(self, tokens, pred_tags):
        """Parse the predict tags to real word
        
        Arguments:
            x {List[str]} -- the origin text
            pred_tags {List[str]} -- predicted tags

        Return:
            entities {List[str]} -- a list of entities
        """
        entities = []
        entity = None
        for idx, st in enumerate(pred_tags):
            if entity is None:
                if st.startswith('B'):
                    entity = {}
                    entity['start'] = idx
                else:
                    continue
            else:
                if st == 'O':
                    entity['end'] = idx
                    name = ''.join(tokens[entity['start']: entity['end']])
                    entities.append(name)
                    entity = None
                elif st.startswith('B'):
                    entity['end'] = idx
                    name = ''.join(tokens[entity['start']: entity['end']])
                    entities.append(name)
                    entity = {}
                    entity['start'] = idx
                else:
                    continue
        return entities


def get_crf_ners(text):
    # text = '罗红霉素和头孢能一起吃吗'
    pred_tags, tokens = crf.predict(text)
    print(pred_tags)
    print(tokens)
    entities = crf.parse(tokens, pred_tags)
    return entities


if __name__ == "__main__":
    crf = CRF(CRF_MODEL_PATH, 'cuda')
    text = '罗红霉素和头孢能一起吃吗'
    print(get_crf_ners(text))
