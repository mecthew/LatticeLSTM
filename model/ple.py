"""
 Author: liujian
 Date: 2021-02-24 22:55:30
 Last Modified by: liujian
 Last Modified time: 2021-02-24 22:55:30
"""

import torch
from torch import nn
import torch.nn.functional as F
import math


class PLE(nn.Module):
    def __init__(self, hidden_size, span_label_size, attr_label_size, dropout_rate=0.3, experts_layers=2, experts_num=1,
                 pactivation='gelu', ple_dropout=0.1, use_ff=False, use_ple_lstm=False):
        """
        Args:
            sequence_encoder (nn.Module): encoder of sequence
            span2id (dict): map from span(et. B, I, O) to id
            attr2id (dict): map from attr(et. PER, LOC, ORG) to id
            compress_seq (bool, optional): whether compress sequence for lstm. Defaults to True.
            share_lstm (bool, optional): whether make span and attr share the same lstm after encoder. Defaults to False.
            span_use_lstm (bool, optional): whether add span lstm layer. Defaults to True.
            span_use_lstm (bool, optional): whether add attr lstm layer. Defaults to False.
            span_use_crf (bool, optional): whether add span crf layer. Defaults to True.
            batch_first (bool, optional): whether fisrt dim is batch. Defaults to True.
            dropout_rate (float, optional): dropout rate. Defaults to 0.3.
            experts_layers(int, optional): layers num of PLE experts. Defaults to 2.
            experts_num(int, optional): experts num of every task. Defaults to 2.
        """
        super(PLE, self).__init__()
        self.mlp_span = nn.Linear(hidden_size, span_label_size)
        self.mlp_attr_start = nn.Linear(hidden_size, attr_label_size)
        self.mlp_attr_end = nn.Linear(hidden_size, attr_label_size)
        if use_ple_lstm:
            self.span_bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
            self.attr_bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=1, bidirectional=True)
        else:
            self.span_bilstm, self.attr_bilstm = None, None
        self.experts_layers = experts_layers
        self.experts_num = experts_num
        self.selector_num = 2
        self.ple_layers = nn.ModuleList()
        for i in range(experts_layers):
            self.ple_layers.append(PLE_Layer(hidden_size, experts_num, self.selector_num, pactivation, ple_dropout, use_ff))
        self.dropout = nn.Dropout(dropout_rate)

    def progressive_layered_extraction(self, gate_shared_output_final, gate_task1_output_final, gate_task2_output_final):
        for ple_layer in self.ple_layers:
            layer_output = ple_layer(gate_shared_output_final, gate_task1_output_final, gate_task2_output_final)
            gate_shared_output_final = gate_shared_output_final + layer_output[0]
            gate_task1_output_final = gate_task1_output_final + layer_output[1]
            gate_task2_output_final = gate_task2_output_final + layer_output[2]
        return gate_shared_output_final, gate_task1_output_final, gate_task2_output_final

    def forward(self, share_hidden, span_hidden, attr_hidden):
        share_rep = share_hidden
        if self.span_bilstm is None:
            span_rep = span_hidden
        else:
            span_rep, _ = self.span_bilstm(span_hidden)
            span_rep = torch.add(*torch.chunk(span_rep, 2, dim=-1))
        if self.attr_bilstm is None:
            attr_rep = attr_hidden
        else:
            attr_rep, _ = self.attr_bilstm(attr_hidden)
            attr_rep = torch.add(*torch.chunk(attr_rep, 2, dim=-1))
        _, span_seqs_hiddens, attr_seqs_hiddens = self.progressive_layered_extraction(share_rep, span_rep, attr_rep)
        # dropout layer
        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens = self.dropout(attr_seqs_hiddens)
        # output layer
        span_logits = self.mlp_span(span_seqs_hiddens) # B, S, V
        attr_start_logits = self.mlp_attr_start(attr_seqs_hiddens) # B, S, V
        attr_end_logits = self.mlp_attr_end(attr_seqs_hiddens) # B, S, V
        
        return span_logits, attr_start_logits, attr_end_logits


class PLE_Layer(nn.Module):
    def __init__(self, hidden_size, experts_num, selector_num, pactivation, ple_dropout, use_ff=False):
        super(PLE_Layer, self).__init__()
        self.pactivation = pactivation
        self.ple_dropout = nn.Dropout(ple_dropout)
        self.use_ff = use_ff

        # experts shared
        self.layers_experts_shared = Linear3D(hidden_size, hidden_size, experts_num)

        # experts task1
        self.layers_experts_task1 = Linear3D(hidden_size, hidden_size, experts_num)

        # experts task2
        self.layers_experts_task2 = Linear3D(hidden_size, hidden_size, experts_num)

        # gates shared
        self.layers_experts_shared_gate = nn.Linear(hidden_size, experts_num * 3)

        # gate task1
        self.layers_experts_task1_gate = nn.Linear(hidden_size, experts_num * selector_num)

        # gate task2
        self.layers_experts_task2_gate = nn.Linear(hidden_size, experts_num * selector_num)

        self.layer_norms = nn.ModuleList()
        for i in range(3):
            self.layer_norms.append(nn.LayerNorm(hidden_size))

        if self.use_ff:
            self.ffs = nn.ModuleList()
            for i in range(3):
                feedforward = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_size * 2, hidden_size)
                )
                self.ffs.append(feedforward)

    def forward(self, gate_shared_output_final, gate_task1_output_final, gate_task2_output_final):
        gate_shared_output_final = self.layer_norms[0](gate_shared_output_final)
        gate_task1_output_final = self.layer_norms[1](gate_task1_output_final)
        gate_task2_output_final = self.layer_norms[2](gate_task2_output_final)

        # shared output
        experts_shared_output = getattr(F, self.pactivation)(self.layers_experts_shared(gate_shared_output_final))
        # task1 output
        experts_task1_output = getattr(F, self.pactivation)(self.layers_experts_task1(gate_task1_output_final))
        # task2 output
        experts_task2_output = getattr(F, self.pactivation)(self.layers_experts_task2(gate_task2_output_final))

        # gate shared output
        gate_shared_output = F.softmax(self.layers_experts_shared_gate(gate_shared_output_final), dim=-1)  # (B, S, C)
        gate_shared_output_final = torch.matmul(gate_shared_output.unsqueeze(-2),
                                                torch.cat(
                                                    [experts_task1_output, experts_shared_output, experts_task2_output],
                                                    dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
        # gate task1 output
        gate_task1_output = F.softmax(self.layers_experts_task1_gate(gate_task1_output_final), dim=-1)  # (B, S, C)
        gate_task1_output_final = torch.matmul(gate_task1_output.unsqueeze(-2),
                                               torch.cat([experts_task1_output, experts_shared_output], dim=-1)
                                               .permute(0, 1, 3, 2)).squeeze(-2)
        # gate task2 output
        gate_task2_output = F.softmax(self.layers_experts_task2_gate(gate_task2_output_final), dim=-1)  # (B, S, C)
        gate_task2_output_final = torch.matmul(gate_task2_output.unsqueeze(-2),
                                               torch.cat([experts_task2_output, experts_shared_output], dim=-1)
                                               .permute(0, 1, 3, 2)).squeeze(-2)

        if self.use_ff:
            gate_shared_output_final = self.ffs[0](gate_shared_output_final)
            gate_task1_output_final = self.ffs[1](gate_task1_output_final)
            gate_task2_output_final = self.ffs[2](gate_task2_output_final)
        gate_shared_output_final = self.ple_dropout(gate_shared_output_final)
        gate_task1_output_final = self.ple_dropout(gate_task1_output_final)
        gate_task2_output_final = self.ple_dropout(gate_task2_output_final)
        return gate_shared_output_final, gate_task1_output_final, gate_task2_output_final


class Linear3D(nn.Module):
    def __init__(self, input_size, output_size1, output_size2):
        """[summary]

        Args:
            input_size (int): first dimension of linear weight.
            output_size1 (int): second dimension of linear weight.
            output_size2 (int): third dimension of linear weight.
        """
        super(Linear3D, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_size, output_size1, output_size2))
        self.bias = nn.Parameter(torch.Tensor(output_size1, output_size2))
        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weight, bias=None):
        """parameters initializtion

        Args:
            weight (torch.Tensor): linear weight.
            bias (torch.Tensor, optional): linear bias. Defaults to None.
        """
        nn.init.xavier_uniform_(weight)
        if bias is not None:
            fan_in = weight.size(0)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)

    def forward(self, input):
        """[summary]

        Args:
            input (torch.Tensor): left matrix of linear projection.

        Returns:
            output (torch.Tensor): output of linear projection.
        """
        output = torch.einsum('...k, kxy -> ...xy', input, self.weight)
        output = torch.add(output, self.bias)
        return output
