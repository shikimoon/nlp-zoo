# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel


class Bert_CRF(nn.Module):
    def __init__(self, tag_to_ix, hidden_dim=768):
        super(Bert_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.transitions = nn.Parameter(torch.randn(
            self.tagset_size, self.tagset_size
        ))
        self.hidden_dim = hidden_dim
        self.start_label_id = self.tag_to_ix['[CLS]']
        self.end_label_id = self.tag_to_ix['[SEP]']
        self.fc = nn.Linear(hidden_dim, self.tagset_size)
        self.bert = BertModel.from_pretrained('bert')

        self.transitions.data[self.start_label_id, :] = -10000
        self.transitions.data[:, self.end_label_id] = -10000
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def _log_sum_exp_batch(self, log_Tensor, axis=-1):
        return torch.max(log_Tensor, axis)[0] + \
               torch.log(
                   torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX 
        '''

        T = feats.shape[1]
        batch_size = feats.shape[0]

        log_alpha = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)  # [batch_size, 1, 16]
        log_alpha[:, 0, self.start_label_id] = 0

        for t in range(1, T):
            log_alpha = (self._log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        log_prob_all_barX = self._log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _score_sentence(self, feats, label_ids):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size, self.tagset_size, self.tagset_size)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(self.device)
        for t in range(1, T):
            score = score + \
                    batch_transitions.gather(-1, (label_ids[:, t] * self.tagset_size + label_ids[:, t - 1]).view(-1, 1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _bert_enc(self, x):
        """
        x: [batchsize, sent_len]
        enc: [batch_size, sent_len, 768]
        """
        with torch.no_grad():
            encoded_layer, _ = self.bert(x)
            enc = encoded_layer[-1]
        return enc

    def _viterbi_decode(self, feats):
        T = feats.shape[1]
        batch_size = feats.shape[0]

        log_delta = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0.

        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.tagset_size), dtype=torch.long)  # psi[0]=0000 useless
        for t in range(1, T):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T - 2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

    def loss(self, sentence, tags):
        feats = self._get_sent_features(sentence)  # [batch_size, max_len, 16]
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.mean(forward_score - gold_score)

    def _get_sent_features(self, sentence):
        """sentence is the ids"""
        embeds = self._bert_enc(sentence)  # [8, 75, 768]
        sent_feats = self.fc(embeds)
        return sent_feats  # [8, 75, 16]

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        sent_feats = self._get_sent_features(sentence)
        score, tag_seq = self._viterbi_decode(sent_feats)
        return score, tag_seq
