# -*- coding: utf-8 -*-
from absa.model.aen import AEN_BERT
from absa.model.bert_spc import BERT_SPC
from absa.model.lcf_bert import LCF_BERT

model_classes = {
    'bert_spc': BERT_SPC,
    'aen_bert': AEN_BERT,
    'lcf_bert': LCF_BERT,
}
