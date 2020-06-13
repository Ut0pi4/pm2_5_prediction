import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Combine(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Combine, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.fc11 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc12 = nn.Linear(hidden_size * 2, output_size)
        self.fc21 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.fc22 = nn.Linear(hidden_size * 2, output_size)
      
      
    def forward(self, hn_1, cn_1, hn_2, cn_2):
        
        hn = torch.cat((hn_1, hn_2), dim=-1)
        cn = torch.cat((cn_1, cn_2), dim=-1)
        
        hn = F.relu(self.fc11(hn))
        hn = F.relu(self.fc12(hn))

        cn = F.relu(self.fc11(cn))
        cn = F.relu(self.fc12(cn))

        return hn, cn

class DownSample(nn.Module):
    def __init__(self, input_length_1, input_length_2, hidden_size):
        super(DownSample, self).__init__()
        
        
        input_size_1 = input_length_1 * hidden_size
        input_size_2 = input_length_2 * hidden_size
        self.input_size_1 = input_size_1
        self.input_size_2 = input_size_2
        self.fc11 = nn.Linear(self.input_size_1, self.input_size_1)
        self.fc12 = nn.Linear(self.input_size_1, hidden_size)
        self.fc21 = nn.Linear(self.input_size_2, self.input_size_2)
        self.fc22 = nn.Linear(self.input_size_2, hidden_size)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        
      
    def forward(self, encoder_outputs_1, encoder_outputs_2):
        encoder_outputs_1 = encoder_outputs_1.view(24, -1)
        
        outputs_1 = F.relu(self.fc11(encoder_outputs_1))
        outputs_1 = F.relu(self.fc12(outputs_1))
         
        encoder_outputs_2 = encoder_outputs_2.view(24, -1)
        outputs_2 = F.relu(self.fc21(encoder_outputs_2))
        outputs_2 = F.relu(self.fc22(outputs_2))
        
        output = torch.cat((outputs_1, outputs_2), -1)
        
        
        output = F.relu(self.fc3(output))
        
        return output

class AttentionRevisedDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(AttentionRevisedDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # input_size = input_size * 35
        # output_size = output_size * 35
        self.input_size = input_size
        self.seq_len = 6
        self.batch_size = 1
        self.output_size = output_size
        self.dropout = nn.Dropout(0.2)

        self.attn = nn.Linear(output_size * 3, self.hidden_size)
        self.attn_combine = nn.Linear(hidden_size + output_size, input_size)

        self.lstm = nn.LSTM(input_size+24, output_size)
        
        self.out_11 = nn.Linear(output_size, hidden_size)
        self.out_12 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hn, cn, encoder_outputs):
        
        
        output = F.one_hot(input.long(), num_classes=6)
        output = output.view(1, 6, -1).float()
        
        stacked_hn = hn.repeat(1, 6, 1)
        stacked_cn = cn.repeat(1, 6, 1)
        
        attn_weights = F.softmax(self.attn(torch.cat((output, stacked_hn, stacked_cn), -1)), -1)
        
        attn_applied = torch.bmm(attn_weights, encoder_outputs.view(1, self.hidden_size, -1))
        
        output = torch.cat((output, attn_applied), -1).view(6, 1, -1)
        
        
        output, hidden = self.lstm(output, (hn, cn))
        output = self.dropout(output)
        
        output = F.relu(self.out_11(output))

        output = F.relu(self.out_12(output))
        output = output.view(-1, 6)
        output = self.softmax(output)
        
        return output, hidden

    def initHidden(self):
        
        return torch.zeros(self.n_layers, self.batch_size, self.output_size, device=device)


class RevisedDecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RevisedDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.input_size = input_size
        self.seq_len = 6
        self.batch_size = 1
        self.output_size = output_size
        self.dropout = nn.Dropout(0.2)
      
        self.lstm = nn.LSTM(input_size, output_size)
        
        self.out_11 = nn.Linear(output_size, hidden_size)
        self.out_12 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hn, cn):
        
        output = F.one_hot(input.long(), num_classes=6)
        output = output.view(6, 1, -1)
        
        
        output, hidden = self.lstm(output.float(), (hn, cn))
        
        output = self.dropout(output)
        
        output = F.relu(self.out_11(output))

        output = F.relu(self.out_12(output))
        output = output.view(-1, 6)
        output = self.softmax(output)
        
        return output, hidden

    def initHidden(self):
        
        return torch.zeros(self.n_layers, self.batch_size, self.output_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_size = input_size
        self.seq_len = 6
        self.batch_size = 1
        self.output_size = output_size
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size, hidden_size)
        
        self.out_11 = nn.Linear(hidden_size, hidden_size)
        self.out_12 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hn, cn):
        
        output = F.one_hot(input.long(), num_classes=6)
        output = output.view(-1, 1, self.input_size)
        output, hidden = self.lstm(output.float(), (hn, cn))
        
        output = self.dropout(output)
        output = output.view(-1, self.hidden_size)
        output = F.relu(self.out_11(output))
        output = self.dropout(output)
        
        output = torch.sigmoid(self.out_12(output))
        
        output = self.softmax(output)
        
        return output, hidden

    def initHidden(self):
        
        return torch.zeros(self.n_layers, self.batch_size, self.hidden_size, device=device)

# Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
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
        
        return output, hidden

    def initHidden(self):

        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)