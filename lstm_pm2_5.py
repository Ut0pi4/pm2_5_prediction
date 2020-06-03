import os
import sys

import numpy as np
import urllib.request
import tensorflow as tf

from pdb import set_trace
import zipfile
import csv

import copy

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOURCE_URL = "https://cloud.tsinghua.edu.cn/d/f519a587a6d943fa9aa0/files/?p=%2F%E5%8C%97%E4%BA%AC%E7%A9%BA%E6%B0%94%E8%B4%A8%E9%87%8F.zip&dl=1"


REMOVED_CSV = {}
REMOVED_CSV["2014"] = {"beijing_all_20141231.csv", "beijing_extra_20141231.csv"}
REMOVED_CSV["2015"] = set()
REMOVED_CSV["2016"] = set()
REMOVED_CSV["2017"] = set()
REMOVED_CSV["2018"] = set()
REMOVED_CSV["2019"] = set()
REMOVED_CSV["2020"] = set()

def maybe_download(filename, work_directory):
    """Download the data from website, unless it's already here."""
    if not tf.io.gfile.exists(work_directory):
        tf.io.gfile.makedirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    # set_trace()
    if not tf.io.gfile.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL, filepath)
    with tf.io.gfile.GFile(filepath) as f:
        print('Successfully downloaded', filename)
    return filepath

def extract_files(filepath):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    # filepath = Path(filepath)
    print('Extracting', filepath)
    #if not tf.io.gfile.exists(filepath):
    #    tf.io.gfile.makedirs(filepath)
        
    with zipfile.ZipFile(filepath+".zip", 'r') as zip_ref:
        #zip_ref.extractall("../air_quality/")
        
        for member in zip_ref.infolist():
            # set_trace()
            
            member.filename = member.filename.encode("cp437").decode("utf8")

            zip_ref.extract(member, "../air_quality/")
#             set_trace()
# #             if member.filename == "北京空气质量"：
# #                 os.rename(member.filename, "pm_2_5_data")
#             set_trace()
    #os.chdir(filepath) # change directory from working dir to dir with files

    for item in os.listdir(filepath): # loop through items in dir
        #set_trace()
        if item.endswith(".zip"): # check for ".zip" extension
            #file_name = os.path.abspath(item) # get full path of files
            zipfile_path = os.path.join(filepath, item)
            zip_ref = zipfile.ZipFile(zipfile_path) # create zipfile object
            zip_ref.extractall(filepath) # extract file to dir
            zip_ref.close() # close file
            #os.remove(file_name) # delete zipped file

def download_and_extract():
    dest = "../air_quality"
    file_name = "北京空气质量.zip"
    filepath = maybe_download(file_name, dest)
    filepath = filepath[:-4]
    extract_files(filepath)


def one_hot(data):
    data_new = copy.deepcopy(data)
#     set_trace()
    data_new[data_new > 250] = -5
    data_new[data_new > 150] = -4
    data_new[data_new > 115] = -3
    data_new[data_new > 75] = -2
    data_new[data_new > 35] = -1
    data_new[data_new > 0] = 0
    data_new = abs(data_new).astype(int)
    
    N, d = data.shape
    one_hot_targets = np.zeros((N, d, 6)) 
    for i, row in enumerate(data_new):
        one_hot_targets[i, :, :] = np.eye(6)[row.reshape(-1)]
#     set_trace()
    
    return one_hot_targets

def process_data(data):
#     data = list(map(int, data))
    N, d = data.shape
    
    new_data = np.zeros((N, d))
#     new_data = []
   
    m = []
    for j, row in enumerate(data):
        row = [float(i) if i != "" else 0 for i in row]
        row = np.array(row)
       
        if np.any(row==0):
#             set_trace()
            index = np.where(row==0)
#             set_trace()
            if len(index[0]) == len(row):
                m.append(j)
                continue
            #for ind in index[0]:
                
            mu = np.sum(row)/(len(row)-len(index[0]))
            row[index[0]] = mu
#             set_trace()
            if mu <= 0:
                set_trace()
#         set_trace()
        new_data[j, :] = row
        
        
    if m:
        for ind in m:    
            new_data[ind] = np.sum(new_data, axis=0)/(len(new_data)-len(m))
    new_data = one_hot(new_data)

    return new_data



def check_data(row, item, year):
    check = [True if i=="" else False for i in row[3:]]
    check_true = True
    if np.all(np.array(check)):
        if item not in REMOVED_CSV[year]:
           
            check_true = False
    return check_true

def add_removed(item, year):
#     REMOVED_CSV[year].add(item)
    split = item.split("_")
    split[-2] = "extra"
    item_1 = "_"
    item_1 = item_1.join(split)
    REMOVED_CSV[year].add(item_1)
    split[-2] = "all"
    item_2 = "_"
    item_2 = item_2.join(split)
    REMOVED_CSV[year].add(item_2)
#     set_trace()
    

def restructure_data(data):
    num_doc = len(data)
    N, d, k = data[0].shape
    
    new_data = np.zeros((num_doc, N, d, k))
    
    for i, dat in enumerate(data):
        new_data[i, :, :, :] = dat
    return new_data

def read_folder(folderpath):

    print("start reading csv in folder: ", folderpath)
    pm2_5s = []
    SO2s = []
    NO2s = []
    COs = []
    O3s = []
    year = folderpath.split("-")[1][:4]
#     set_trace()
    
    for item in os.listdir(folderpath):
        if "beijing" not in item:
            continue
        if item in REMOVED_CSV[year]:
            continue
        
        filepath_csv = folderpath + "/" + item
#         print(filepath_csv)
        
        pm2_5 = []
        SO2 = []
        NO2 = []
        CO = []
        O3 = []
        d = 0
        
        with open(filepath_csv, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
    #         set_trace()
            prev_header = []
            for i, row in enumerate(csv_reader):
#                 set_trace()
                try:
                    row[2]=="SO2"
                        
                except:
                    if item not in REMOVED_CSV[year]:
                        add_removed(item, year)
#                         REMOVED_CSV[year].add(item)
                    break
                
                if len(row[3:]) != 35:
                    if item not in REMOVED_CSV[year]:
                        add_removed(item, year)
#                         REMOVED_CSV[year].add(item)
                    break

               
                    
                if i==0:
                    header = row[3:]
                    if prev_header:
                        if header != prev_header:
                            set_trace()
                            break
                    prev_header = header
                    continue
                else:
                    if row[2] in ["PM2.5", "SO2", "NO2", "CO", "O3"]:
                        if not check_data(row[3:], item, year):
                            add_removed(item, year)
#                             REMOVED_CSV[year].add(item)
                            break
                        else:
                            if row[2]=="PM2.5":
                                pm2_5.append(row[3:])
                            elif row[2]=="SO2":
                                SO2.append(row[3:])
                            elif row[2]=="NO2":
                                NO2.append(row[3:])
                            elif row[2]=="CO":
                                CO.append(row[3:])
                            elif row[2]=="O3":
                                O3.append(row[3:])

        split = filepath_csv.split("_")
        if split[-2] == "all":
            if len(pm2_5) != 24:
                add_removed(item, year)
#                 REMOVED_CSV[year].add(item)
                continue
        elif split[-2] == "extra":
#             set_trace()
            if len(SO2) != 24:
                add_removed(item, year)
#                 REMOVED_CSV[year].add(item)
                continue
            if len(NO2) != 24:
                add_removed(item, year)
#                 REMOVED_CSV[year].add(item)
                continue
            if len(CO) != 24:
                add_removed(item, year)
#                 REMOVED_CSV[year].add(item)
                continue
            if len(O3) != 24:
                add_removed(item, year)
#                 REMOVED_CSV[year].add(item)
                continue
                        

        if pm2_5:
            pm2_5 = np.array(pm2_5).T
            pm2_5 = process_data(pm2_5)
            pm2_5s.append(pm2_5)

        if SO2:
            SO2 = np.array(SO2).T
            SO2 = process_data(SO2)
            SO2s.append(SO2)

        if NO2:
            NO2 = np.array(NO2).T
            NO2 = process_data(NO2)
            NO2s.append(NO2)

        if CO:
            CO = np.array(CO).T
            CO = process_data(CO)
            COs.append(CO)

        if O3:
            O3 = np.array(O3).T
            O3 = process_data(O3)
            O3s.append(O3)
    
    pm2_5s = restructure_data(pm2_5s)
    SO2s = restructure_data(SO2s)
    NO2s = restructure_data(NO2s)
    COs = restructure_data(COs)
    O3s = restructure_data(O3s)
    
    print("complete read folder")
    return pm2_5s, SO2s, NO2s, COs, O3s
    


# download_and_extract()
# if not tf.io.gfile.exists("C:/Users/xiang/Desktop/Project/PR/PM2_5/air_quality/pm2_5_dataset"):
# #         print(tf.io.gfile.exists("C:/Users/xiang/Desktop/Project/PR/PM2_5/air_quality/北京空气质量/"))
#     try:
#         os.rename("C:/Users/xiang/Desktop/Project/PR/PM2_5/air_quality/北京空气质量", "C:/Users/xiang/Desktop/Project/PR/PM2_5/air_quality/pm2_5_dataset")
#     except:
#         print("unable to change name")
folderpath = "../air_quality/北京空气质量"
for key, values in REMOVED_CSV.items():
    print("length of REMOVED_CSV in year %s: %d" %(key, len(values)))
print("")  

pm2_5s_years = []
SO2s_years = []
NO2s_years = []
COs_years = []
O3s_years = []
if tf.io.gfile.exists(folderpath):
    for item in os.listdir(folderpath):
        if ".zip" not in item and "beijing" in item:
            pm2_5s, SO2s, NO2s, COs, O3s = read_folder(folderpath+"/"+item)
            
            pm2_5s_years.append(pm2_5s)
            SO2s_years.append(SO2s)
            NO2s_years.append(NO2s)
            COs_years.append(COs)
            O3s_years.append(O3s)
            
            

for key, values in REMOVED_CSV.items():
    print("length of REMOVED_CSV in year %s: %d" %(key, len(values)))
print("")            

print("Start checking REMOVED_CSV pairs")
for key, values in REMOVED_CSV.items():
    for value_a in values:
        two = 1
        split = value_a.split("_")
        for value_b in values:
            if value_a != value_b:
                if split[-1] in value_b:
                    two += 1
                    break                    
        if two != 2:
            set_trace()
print("Complete check")
#     set_trace()

import random

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
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
        
#         encoder_outputs[ei, :, :] = encoder_output.view(24, encoder.hidden_size)
#         set_trace()

#     decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_input = target_tensor[0, 0, :]
    set_trace()
#     decoder_hidden = (hn, cn)

#     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = True
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            for t in range(len(target_tensor[di])-6):
                set_trace()
                decoder_output, (hn, cn) = decoder(
                    decoder_input, hn, cn)
                set_trace()

                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, hn, cn)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)



def reshape_data(data, i):
    data1 = data[:i, :, :, :]
    data2 = data[i+1:, :, :, :]
    new_data = np.concatenate((data1, data2), axis=0)
    new_data = new_data.reshape((-1, 24, 6))
    return new_data

def save_checkpoint(epoch, encoder, decoder, encoder_optimizer, decoder_optimizer):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    """
    state = {'epoch': epoch,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = '../checkpoint_lstm.pth.tar'
    torch.save(state, filename)

def trainIters(year_data, encoder, decoder, encoder_optimizer, decoder_optimizer, 
               epochs, print_every=1, plot_every=1, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

#     training_pairs = [tensorsFromPair(random.choice(pairs))
#                       for i in range(n_iters)]
    criterion = nn.NLLLoss()
#     set_trace()
    G, b, m, k = year_data.shape
    
    for epoch in range(epochs):
        for i in range(G):
            input_tensor = reshape_data(year_data, i)
            input_tensor = torch.from_numpy(input_tensor).cuda()
            target_tensor = year_data[i, :, :, :].reshape((-1, 24, 6))
            target_tensor = torch.from_numpy(target_tensor).cuda()
#             set_trace()
            loss = train(input_tensor, target_tensor, encoder,
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
#     for iter in range(1, n_iters + 1):
#         training_pair = training_pairs[iter - 1]
#         input_tensor = training_pair[0]
#         target_tensor = training_pair[1]

#         loss = train(input_tensor, target_tensor, encoder,
#                      decoder, encoder_optimizer, decoder_optimizer, criterion)
#         print_loss_total += loss
#         plot_loss_total += loss

#         if iter % print_every == 0:
#             print_loss_avg = print_loss_total / print_every
#             print_loss_total = 0
#             print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
#                                          iter, iter / n_iters * 100, print_loss_avg))

#         if iter % plot_every == 0:
#             plot_loss_avg = plot_loss_total / plot_every
#             plot_losses.append(plot_loss_avg)
#             plot_loss_total = 0

    showPlot(plot_losses)




import os

class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_size = input_size
#         self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hn, cn):
        output = input.view(1, 1, self.input_size)
        output = F.relu(output)
#         set_trace()
        output, hidden = self.lstm(output, (hn, cn))
        output = self.softmax(self.out(output[-1, :, :]))
        set_trace()
        return output, hidden

    def initHidden(self):
        
        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size

#         self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input, hn, cn):
#         embedded = self.embedding(input).view(1, 1, -1)
#         output = embedded
        output = input.view(24,1,6)
        
        output, hidden = self.lstm(output.float(), (hn, cn))
#         set_trace()
        return output, hidden

    def initHidden(self):

        return torch.zeros(self.n_layers, 1, self.hidden_size, device=device)


hidden_size = 256
input_size = 6
output_size = 6
epochs = 10
learning_rate = 0.001

checkpoint = '../checkpoint_lstm.pth.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(checkpoint):
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint["encoder"]
    decoder = checkpoint["decoder"]
    epoch = checkpoint["epoch"] + 1
    encoder_optimizer = checkpoint["encoder_optimizer"]
    decoder_optimizer = checkpoint["decoder_optimizer"]
else:
    epoch = 0
    encoder = EncoderRNN(input_size, hidden_size).cuda()
    # attn_decoder1 = AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)
    decoder = DecoderRNN(input_size, hidden_size, output_size).cuda()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
if epoch < epochs:
    trainIters(pm2_5s_years[1], encoder, decoder, encoder_optimizer, decoder_optimizer, epochs)

print("finish training")