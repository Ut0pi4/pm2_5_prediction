import random

import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from model import EncoderRNN, DecoderRNN
from preprocessing import preprocess

teacher_forcing_ratio = 0.5

def train_one_epoch(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    hn = encoder.initHidden()
    cn = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

#     encoder_outputs = torch.zeros(input_length, 24, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, (hn, cn) = encoder(
            input_tensor[ei], hn, cn)
        
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    decoder_input = target_tensor[0, :]
           
    for di in range(target_length):
#         set_trace()
        decoder_output, (hn, cn) = decoder(decoder_input, hn, cn) 
        loss += criterion(decoder_output, target_tensor[di, :].view(-1,).long())

        if use_teacher_forcing:
            decoder_input = target_tensor[di, :]
        else:
            decoder_output = decoder_output.view(6, 35 ,6)
            topv, topi = decoder_output.topk(1, dim=2)
            decoder_input = topi.squeeze().detach()
#         print(decoder_output.shape)
#         print(decoder_input.shape)
#         set_trace()
        
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
#     set_trace()
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))




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
#     set_trace()
#     new_data = new_data.reshape((-1, 24, 34))
    return new_data

def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer， input_type):
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
    filename = '../checkpoint_lstm.pth_'+input_type+'.tar'
    torch.save(state, filename)

    
def train(data_e, data_d, encoder, decoder, encoder_optimizer, decoder_optimizer, 
               start_epoch, epochs, print_every=1, plot_every=1, learning_rate=0.01,
               input_type=0):
    encoder.train()
    decoder.train()
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

#     training_pairs = [tensorsFromPair(random.choice(pairs))
#                       for i in range(n_iters)]
    criterion = nn.NLLLoss()
#     g, t, f= year_data.shape
    
    input_tensor = data_e
    input_tensor = torch.from_numpy(input_tensor).to(device)
    target_tensor = data_d
    target_tensor = torch.from_numpy(target_tensor).to(device)
    
    for epoch in range(start_epoch, epochs+1):

        loss = train_one_epoch(input_tensor, target_tensor, encoder,
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
            
        save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, input_type)


    showPlot(plot_losses)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PM2.5 Prediction")
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--dest', type=str, default="../air_quality", help='path to dataset.')
    parser.add_argument('--epochs', type=int, default=10, help='epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--input_type', type=int, default=0, help='0 or 1, choose which input type to run')
    
    
    # parser.add_argument('--img-path', type=str, help='path to your image.')
    # parser.add_argument('--video-path', type=str, default='0', help='path to your video, `0` means to use camera.')
    # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    args = parser.parse_args()

    hidden_size = args.hidden_size
    input_size_enc = 35
    input_size_dec = 35
    output_size = 35
    epochs = args.epochs
    learning_rate = args.lr

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    file_name = "北京空气质量.zip"
    dest = args.dest
    
    feature_data, pm2_5s_1, pm2_5s_2, pm2_5s_1_2014, pm2_5s_2_2014 = preprocess(file_name, dest)

    input_type = args.input_type
    if input_type == 0:
        data_e = feature_data
        data_d = pm2_5s_2[:, :6, :]
    else:
        data_e = pm2_5s_1[:, :24, :]
        data_d = pm2_5s_1[:, 24:24+6, :]

    checkpoint = '../checkpoint_lstm.pth_'+input_type+'.tar'
    encoder = EncoderRNN(input_size_enc, hidden_size).to(device)
    decoder = DecoderRNN(input_size_dec, hidden_size, output_size).to(device)

    # if os.path.exists(checkpoint):
    #     checkpoint = torch.load(checkpoint)    
    #     encoder.load_state_dict(checkpoint["encoder"])
    #     decoder.load_state_dict(checkpoint["decoder"])
        
    #     epoch = checkpoint["epoch"]+1
    #     encoder_optimizer = checkpoint["encoder_optimizer"]
    #     decoder_optimizer = checkpoint["decoder_optimizer"]
    # else:
    epoch = 1
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    if epoch <= epochs:
        print("start training from epoch %d" %epoch)
        train(data_e, data_d, encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, epochs)
    else:
        print("epoch trained (%d) exceeds maximum epochs (%d)" %(epoch, epochs))
    print("finish training")