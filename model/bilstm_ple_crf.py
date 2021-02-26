# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-05 23:15:17

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bilstm import BiLSTM
from crf import CRF
from ple import PLE

class BiLSTM_CRF(nn.Module):
    def __init__(self, data):
        super(BiLSTM_CRF, self).__init__()
        print "build batched lstmcrf..."
        self.gpu = data.HP_gpu
        ## add two more label for downlayer lstm, use original label size for CRF
        span_label_size = data.span_label_size
        data.span_label_size += 2
        print(data.span_label_alphabet.instances)
        print(data.attr_label_alphabet.instances)
        self.lstm = BiLSTM(data)
        self.span_crf = CRF(span_label_size, self.gpu)
        self.ple = PLE(hidden_size=data.HP_hidden_dim, span_label_size=data.span_label_size, attr_label_size=data.attr_label_size, dropout_rate=0.3, experts_layers=2, experts_num=2)
        if self.gpu:
            self.ple = self.ple.cuda()
        # ce loss
        self.attr_criterion = nn.CrossEntropyLoss() # reduction='none'


    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, span_labels, attr_start_labels, attr_end_labels, mask):
        span_logits, attr_start_logits, attr_end_logits = self.forward(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        inputs_seq_len = mask.sum(dim=-1).float()
        span_loss = self.span_crf.neg_log_likelihood_loss(span_logits, mask, span_labels)
        scores, span_tag_seq = self.span_crf._viterbi_decode(span_logits, mask)
        attr_start_loss = self.attr_criterion(attr_start_logits.permute(0, 2, 1).unsqueeze(-1), attr_start_labels.unsqueeze(-1)).squeeze(-1) # B * S
        attr_start_loss = torch.sum(attr_startss * mask.float(), dim=-1).float() / inputs_seq_len # B
        attr_end_loss = self.attr_criterion(attr_end_logits.permute(0, 2, 1).unsqueeze(-1), attr_end_labels.unsqueeze(-1)).squeeze(-1)  # B * S
        attr_end_loss = torch.sum(attr_end_loss * mask.float(), dim=-1).float() / inputs_seq_len # B
        attr_start_loss, attr_end_loss = attr_start_loss.mean(), attr_end_loss.mean()
        total_loss = (span_loss + attr_start_loss + attr_end_loss) / 3
        _, attr_start_output = torch.max(attr_start_logits, dim=-1)
        _, attr_end_output = torch.max(attr_end_logits, dim=-1)
        return total_loss, span_tag_seq, attr_start_output, attr_end_output


    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        hidden = self.get_lstm_features(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        span_logits, attr_start_logits, attr_end_logits = self.ple(hidden)
        return span_logits, attr_start_logits, attr_end_logits


    def get_lstm_features(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover):
        return self.lstm.get_lstm_features(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
