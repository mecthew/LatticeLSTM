# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2017-10-17 16:47:32
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-01-05 23:15:17

import torch
import torch.nn as nn

from .bilstm import BiLSTM
from .crf import CRF
from .ple import PLE


class BiLSTM_CRF(nn.Module):
    def __init__(self, data, use_ple_lstm):
        super(BiLSTM_CRF, self).__init__()
        print("build batched lstmcrf...")
        self.gpu = data.HP_gpu
        ## add two more label for downlayer lstm, use original label size for CRF
        span_label_size = data.span_label_size
        data.span_label_size += 2
        print((data.span_label_alphabet.instances))
        print((data.attr_label_alphabet.instances))
        self.latticelstm_num = data.latticelstm_num
        self.lstm_list = nn.ModuleList()
        for i in range(data.latticelstm_num):
            self.lstm_list.append(BiLSTM(data))
        self.span_crf = CRF(span_label_size, self.gpu)
        self.ple = PLE(hidden_size=data.HP_hidden_dim, span_label_size=data.span_label_size,
                       attr_label_size=data.attr_label_size, dropout_rate=0.3, experts_layers=2,
                       experts_num=1, ple_dropout=0.1, use_ple_lstm=use_ple_lstm)
        # ce loss
        weight = torch.FloatTensor(
            [1.0 if i != data.attr_label_alphabet.get_index('') else 0.1 for i in range(data.attr_label_size)])
        print("attr weight: {}".format(weight))
        self.attr_criterion = nn.CrossEntropyLoss(reduction='none', weight=weight)

    def neg_log_likelihood_loss(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                char_seq_lengths, char_seq_recover, span_labels, attr_start_labels, attr_end_labels,
                                mask):
        shape = word_inputs.size()
        span_logits, attr_start_logits, attr_end_logits = self.forward(gaz_list, word_inputs, biword_inputs,
                                                                       word_seq_lengths, char_inputs, char_seq_lengths,
                                                                       char_seq_recover)
        inputs_seq_len = mask.sum(dim=-1).float()
        span_loss = self.span_crf.neg_log_likelihood_loss(span_logits, mask, span_labels)
        scores, span_tag_seq = self.span_crf._viterbi_decode(span_logits, mask)
        attr_start_loss = self.attr_criterion(attr_start_logits.permute(0, 2, 1), attr_start_labels)  # B * S
        attr_start_loss = torch.sum(attr_start_loss * mask.float(), dim=-1).float() / inputs_seq_len  # B
        attr_end_loss = self.attr_criterion(attr_end_logits.permute(0, 2, 1), attr_end_labels)  # B * S
        attr_end_loss = torch.sum(attr_end_loss * mask.float(), dim=-1).float() / inputs_seq_len  # B
        attr_start_loss, attr_end_loss = attr_start_loss.mean(), attr_end_loss.mean()
        total_loss = (0.1 * span_loss + attr_start_loss + attr_end_loss) / 3
        # print(("=="*20, "loss"))
        # print((span_loss, attr_start_loss, attr_end_loss))
        _, attr_start_output = torch.max(attr_start_logits, dim=-1)
        _, attr_end_output = torch.max(attr_end_logits, dim=-1)
        # print(attr_start_output)
        # print(attr_start_labels)
        return total_loss, span_tag_seq, attr_start_output, attr_end_output

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                char_seq_recover):
        encoder_hiddens = []
        for i in range(self.latticelstm_num):
            hidden = self.lstm_list[i].get_lstm_features(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                                       char_seq_lengths, char_seq_recover)
            encoder_hiddens.append(hidden)
        if self.latticelstm_num == 1:
            encoder_hiddens = encoder_hiddens * 3   # 重复三次
        elif self.latticelstm_num == 2:
            encoder_hiddens = encoder_hiddens + [encoder_hiddens[-1]]   # 重复最后一个
        span_logits, attr_start_logits, attr_end_logits = self.ple(*encoder_hiddens)
        return span_logits, attr_start_logits, attr_end_logits

    def get_lstm_features(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                          char_seq_recover):
        return self.lstm.get_lstm_features(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                           char_seq_lengths, char_seq_recover)
