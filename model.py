import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class AttnDecoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, n_layers=1):
#         super(AttnDecoderRNN, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.n_layers = n_layers
#         self.input_size = input_size
#         self.seq_len = 6
#         self.output_size = output_size
        
#         self.attn = nn.Linear(self.hidden_size * 2, )
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.lstm = nn.LSTM(input_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.out_1 = nn.Linear(1, 6)
#         self.softmax = nn.LogSoftmax(dim=2)

#     def forward(self, input, hn, cn, encoder_outputs):
#         output = input.view(self.seq_len, 1, self.input_size)

#         attn_weights = F.softmax(
#             self.attn(torch.cat(()))
#         )

#         output = F.relu(output)
#         output, hidden = self.lstm(output.float(), (hn, cn))
#         output = output.view(self.seq_len, self.hidden_size)
        
#         output = self.out(output).view(self.seq_len, self.output_size, 1)
#         output = self.out_1(output)
#         output = self.softmax(output)
#         output = output.view(-1, 6)
# #         set_trace()
# #         output = output.view(self.seq_len, self.input_size)
#         return output, hidden

#     def initHidden(self):
        
#         return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_size = input_size
        self.seq_len = 6
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.out_1 = nn.Linear(1, 6)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hn, cn):
        output = input.view(self.seq_len, 1, self.input_size)
        output = F.relu(output)
        output, hidden = self.lstm(output.float(), (hn, cn))
        output = output.view(self.seq_len, self.hidden_size)
        
        output = self.out(output).view(self.seq_len, self.output_size, 1)
        output = self.out_1(output)
        output = self.softmax(output)
        output = output.view(-1, 6)
#         set_trace()
#         output = output.view(self.seq_len, self.input_size)
        return output, hidden

    def initHidden(self):
        
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.seq_len = 24
        self.n_layers = n_layers
        
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input, hn, cn):
        
        output = input.view(self.seq_len, 1, self.input_size)
        output, hidden = self.lstm(output.float(), (hn, cn))
        
#         set_trace()
        return output, hidden

    def initHidden(self):

        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)

