"""
 Author: liujian
 Date: 2021-02-24 22:55:30
 Last Modified by: liujian
 Last Modified time: 2021-02-24 22:55:30
"""

import torch
from torch import nn
import torch.nn.functional as F


class PLE(nn.Module):
    def __init__(self, hidden_size, span_label_size, attr_label_size, dropout_rate=0.3, experts_layers=2, experts_num=2):
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
        self.experts_layers = experts_layers
        self.experts_num = experts_num
        self.selector_num = 2
        self.layers_experts_shared = nn.ModuleList()
        self.layers_experts_task1 = nn.ModuleList()
        self.layers_experts_task2 = nn.ModuleList()
        self.layers_experts_shared_gate = nn.ModuleList()
        self.layers_experts_task1_gate = nn.ModuleList()
        self.layers_experts_task2_gate = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(experts_layers):
            # experts shared
            self.layers_experts_shared.append(nn.Linear(hidden_size, hidden_size * experts_num))        

            # experts task1
            self.layers_experts_task1.append(nn.Linear(hidden_size, hidden_size * experts_num))            

            # experts task2
            self.layers_experts_task2.append(nn.Linear(hidden_size, hidden_size * experts_num))       

            # gates shared
            self.layers_experts_shared_gate.append(nn.Linear(hidden_size, experts_num * 3))            

            # gate task1
            self.layers_experts_task1_gate.append(nn.Linear(hidden_size, experts_num * self.selector_num))            

            # gate task2
            self.layers_experts_task2_gate.append(nn.Linear(hidden_size, experts_num * self.selector_num))            


    def progressive_layered_extraction(self, gate_shared_output_final, gate_task1_output_final, gate_task2_output_final):
        shape = gate_shared_output_final.size()
        for i in range(self.experts_layers):
            # shared  output
            experts_shared_output = F.relu(self.layers_experts_shared[i](gate_shared_output_final))
            experts_shared_output = experts_shared_output.contiguous().view(*(shape + (self.experts_num, )))

            # task1 output
            experts_task1_output = F.relu(self.layers_experts_task1[i](gate_task1_output_final))
            experts_task1_output = experts_task1_output.contiguous().view(*(shape + (self.experts_num, )))

            # task2 output
            experts_task2_output = F.relu(self.layers_experts_task2[i](gate_task2_output_final))
            experts_task2_output = experts_task2_output.contiguous().view(*(shape + (self.experts_num, )))

            # gate shared output
            gate_shared_output = self.layers_experts_shared_gate[i](gate_shared_output_final) # (B, S, C)
            gate_shared_output = F.softmax(gate_shared_output, dim=-1)
            gate_shared_output = torch.matmul(gate_shared_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output, experts_task2_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_shared_output = F.dropout(gate_shared_output, p=0.1)
            gate_shared_output_final = gate_shared_output
            # gate_shared_output_final = self.layers_layernorm[i][0](gate_shared_output + gate_shared_output_final)

            # gate task1 output
            gate_task1_output = self.layers_experts_task1_gate[i](gate_task1_output_final) # (B, S, C)
            gate_task1_output = F.softmax(gate_task1_output, dim=-1)
            gate_task1_output = torch.matmul(gate_task1_output.unsqueeze(-2), 
                                    torch.cat([experts_task1_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_task1_output = F.dropout(gate_task1_output, p=0.1)
            gate_task1_output_final = gate_task1_output
            # gate_task1_output_final = self.layers_layernorm[i][1](gate_task1_output + gate_task1_output_final)

            # gate task2 output
            gate_task2_output = self.layers_experts_task2_gate[i](gate_task2_output_final) # (B, S, C)
            gate_task2_output = F.softmax(gate_task2_output, dim=-1)
            gate_task2_output = torch.matmul(gate_task2_output.unsqueeze(-2), 
                                    torch.cat([experts_task2_output, experts_shared_output], dim=-1).permute(0, 1, 3, 2)).squeeze(-2)
            # gate_task2_output = F.dropout(gate_task2_output, p=0.1)
            gate_task2_output_final = gate_task2_output
            # gate_task2_output_final = self.layers_layernorm[i][2](gate_task2_output + gate_task2_output_final)
        
        return gate_shared_output_final, gate_task1_output_final, gate_task2_output_final


    def forward(self, share_hidden, span_hidden, attr_hidden):
        _, span_seqs_hiddens, attr_seqs_hiddens = self.progressive_layered_extraction(share_hidden, span_hidden, attr_hidden)
        # dropout layer
        span_seqs_hiddens = self.dropout(span_seqs_hiddens)
        attr_seqs_hiddens = self.dropout(attr_seqs_hiddens)
        # output layer
        span_logits = self.mlp_span(span_seqs_hiddens) # B, S, V
        attr_start_logits = self.mlp_attr_start(attr_seqs_hiddens) # B, S, V
        attr_end_logits = self.mlp_attr_end(attr_seqs_hiddens) # B, S, V
        
        return span_logits, attr_start_logits, attr_end_logits
