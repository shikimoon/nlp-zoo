import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        global_context_indices, global_segments_indices = inputs[0], inputs[1]
        _, pooled_output = self.bert(global_context_indices, token_type_ids=global_segments_indices)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
