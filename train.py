import random

import time
import math
import os

import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from model import *
from preprocessing import preprocess

teacher_forcing_ratio = 0.5

# Taken from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Taken from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def timeSince(since, percent):
    now = time.time()
    s = now - since
#     set_trace()
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# Taken from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)



def reshape_data(data, i):
    data1 = data[:, :, :i]
    data2 = data[:, :, i+1:]
    new_data = np.concatenate((data1, data2), axis=2)
    return new_data

# Taken from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'encoder': encoder.state_dict(),
             'decoder': decoder.state_dict(),
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = '../checkpoint_lstm.pth.tar'
    torch.save(state, filename)

# Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def train_one_epoch(input_tensor_1, input_tensor_2, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    hn_1 = encoder.initHidden()
    cn_1 = encoder.initHidden()
    hn_2 = encoder.initHidden()
    cn_2 = encoder.initHidden()
    # hn = decoder.initHidden()
    # cn = decoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length_1 = input_tensor_1.size(0)
    input_length_2 = input_tensor_2.size(0)
    
    target_length = target_tensor.size(0)

    encoder_outputs_1 = torch.zeros(input_length_1, 24, encoder.hidden_size, device=device)
    encoder_outputs_2 = torch.zeros(input_length_2, 24, encoder.hidden_size, device=device)

    loss = 0
    
    for ei in range(input_tensor_1.size(0)):
        encoder_output_1, (hn_1, cn_1) = encoder(input_tensor_1[ei], hn_1, cn_1)
        encoder_outputs_1[ei] = encoder_output_1.detach().view(24, encoder.hidden_size)
    for ei in range(input_tensor_2.size(0)):
        encoder_output_2, (hn_2, cn_2) = encoder(input_tensor_2[ei], hn_2, cn_2)
        encoder_outputs_2[ei] = encoder_output_2.detach().view(24, encoder.hidden_size)
    
    # ds = DownSample(input_length_1, input_length_2, encoder.hidden_size).to(device)

    # outputs = ds(encoder_outputs_1, encoder_outputs_2)
    
    combine = Combine(encoder.hidden_size, decoder.output_size).to(device)
    
    hn, cn = combine(hn_1, cn_1, hn_2, cn_2)
    hn = hn.repeat(1, decoder.batch_size, 1)
    cn = cn.repeat(1, decoder.batch_size, 1)
    decoder_input = target_tensor[0, :]


    for di in range(target_length):
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        # decoder_output, (hn, cn) = decoder(decoder_input, hn, cn, outputs) 
        decoder_output, (hn, cn) = decoder(decoder_input, hn, cn) 
        
        loss += criterion(decoder_output, target_tensor[di, :].view(-1,).long())
  
        if use_teacher_forcing:
            decoder_input = target_tensor[di, :]
        else:
            decoder_output = decoder_output.view(6, 35 ,6)
            topv, topi = decoder_output.topk(1, dim=2)
            decoder_input = topi.squeeze().detach()
        # set_trace()    
        
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

# Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def train(data_e_1, data_e_2, data_d, encoder, decoder, encoder_optimizer, decoder_optimizer, 
               start_epoch, epochs, weights=[0.05, 0.05, 1, 3, 3, 5], print_every=1, 
               plot_every=1, learning_rate=0.01):
    encoder.train()
    decoder.train()
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_everyNone

    if len(weights) != 6:
        print("wrong weights size: need 6 inputs")
        return None
    weights = torch.FloatTensor(weights).to(device)
    criterion = nn.NLLLoss(weight=weights)

    input_tensor_1 = data_e_1
    input_tensor_1 = torch.from_numpy(input_tensor_1).to(device)
    input_tensor_2 = data_e_2
    input_tensor_2 = torch.from_numpy(input_tensor_2).to(device)
    target_tensor = data_d
    target_tensor = torch.from_numpy(target_tensor).to(device)
    
    batch_loss = []
    for epoch in range(start_epoch, epochs+1):
  
        loss = train_one_epoch(input_tensor_1, input_tensor_2, target_tensor, encoder,
                 decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / epochs),
                                         epoch, epoch / epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            
        save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer)


    showPlot(plot_losses)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PM2.5 Prediction")
    
    parser.add_argument('--dest', type=str, default="../air_quality", help='path to dataset.')
    parser.add_argument('--epochs', type=int, default=50, help='epochs to train')
    parser.add_argument('--weights', type=list, default=[0.05, 0.05, 1, 3, 3, 5], help='weights to train')
    parser.add_argument('--checkpoint', type=str, default='../checkpoint_lstm.pth.tar', help="path to checkpoint")


    args = parser.parse_args()

    hidden_size = 64
    input_size_enc = 35
    input_size_dec = 6 * 35
    output_size = 6 * 35
    epochs = args.epochs
    learning_rate = 0.001
    weights = args.weights
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file_name = "北京空气质量.zip"
    dest = args.dest
    
    feature_data, pm2_5s = preprocess(file_name, dest)

    
    train_set = concat_years(pm2_5s[1:])
    data_e_1 = concat_years(feature_data[1:])

    data_e_2 = train_set[:, :24, :]
    data_d = train_set[:, 24:30, :]

    checkpoint = args.checkpoint
    encoder = EncoderRNN(input_size_enc, hidden_size).to(device)
    decoder = RevisedDecoderRNN(input_size_dec, hidden_size, output_size).to(device)

    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint)    
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        
        epoch = checkpoint["epoch"]+1
        encoder_optimizer = checkpoint["encoder_optimizer"]
        decoder_optimizer = checkpoint["decoder_optimizer"]
    else:
        epoch = 1
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    if epoch <= epochs:
        print("start training from epoch %d" %epoch)
        train(data_e_1, data_e_2, data_d, encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, epochs)
    else:
        print("epoch trained (%d) exceeds maximum epochs (%d)" %(epoch, epochs))
    print("finish training")