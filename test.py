
import os
import numpy as np
from preprocessing import preprocess
from model import *
import torch
import torch.nn as nn
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(pred, target):

    pred = pred.squeeze()
    np_pred = pred.cpu().numpy()
    np_target = target.cpu().numpy()
    d = pred.shape
    accu = np.sum(np_pred==np_target)/(d)
    counts_pred = np.zeros((6))
    counts_target = np.zeros((6))

    for i in range(6):
        counts_target[i] = np.sum(np_target==i)
        index = np.where(np_target==i)
        counts_pred[i] = np.sum(np_pred[index]==i)
    return accu, counts_pred, counts_target

# Modified from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def evaluate(input_tensor_1, input_tensor_2, target_tensor, encoder, decoder):
    encoder.eval()
    decoder.eval()
    
    accu_all = []
    
    losses = 0
    criterion = nn.NLLLoss()
    
    target_length = target_tensor.size(0)
    counts_p_all = np.zeros((6))
    counts_t_all = np.zeros((6))
    batch_loss = []
    with torch.no_grad():
        hn_1 = encoder.initHidden()
        cn_1 = encoder.initHidden()
        hn_2 = encoder.initHidden()
        cn_2 = encoder.initHidden()
        hn = decoder.initHidden()
        cn = decoder.initHidden() 
        
        for ei in range(input_tensor_1.size(0)):
            _, (hn_1, cn_1) = encoder(
                input_tensor_1[ei], hn_1, cn_1)
        
        for ei in range(input_tensor_2.size(0)):
            _, (hn_2, cn_2) = encoder(
                input_tensor_2[ei], hn_2, cn_2)

        
        combine = Combine(encoder.hidden_size, decoder.output_size).to(device)

        hn, cn = combine(hn_1, cn_1, hn_2, cn_2)
        hn = hn.repeat(1, decoder.batch_size, 1)
        cn = cn.repeat(1, decoder.batch_size, 1)
        
        decoder_input = target_tensor[0, :6]    
        
              
        for di in range(target_length):
            
            decoder_output, (hn, cn) = decoder(decoder_input, hn, cn)  
            loss = criterion(decoder_output, target_tensor[di, :].view(-1,).long())
            losses += loss
            batch_loss.append(loss.item())
            decoder_output = decoder_output.view(6, 35 ,6)
            topv, topi = decoder_output.topk(1, dim=2)
            accu, pred_counts, target_counts = accuracy(topi.view(-1), target_tensor[di, :].view(-1))
            counts_p_all += pred_counts
            counts_t_all += target_counts
            accu_all.append(accu)
            decoder_input = topi.squeeze().detach()
            
#     set_trace()
    # x = np.arange(len(accu_all))   
    # plt.plot(x, accu_all)
    # plt.ylabel("accuracy")
    # plt.show()

    # x = np.arange(len(batch_loss))   
    # plt.plot(x, batch_loss)
    # plt.ylabel("loss")
    # plt.show()
    return losses.item() / target_length, np.mean(accu_all), accu_all, counts_p_all, counts_t_all

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="PM2.5 Prediction")
    
    parser.add_argument('--dest', type=str, default="../air_quality", help='path to dataset.')
    parser.add_argument('--epochs', type=int, default=10, help='epochs to train')
    parser.add_argument('--checkpoint', type=str, default='../checkpoint_lstm.pth.tar', help="path to checkpoint")
    args = parser.parse_args()

    hidden_size = 64
    input_size_enc = 35
    input_size_dec = 6 * 35
    output_size = 6 * 35
    epochs = args.epochs
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    file_name = "北京空气质量.zip"
    dest = args.dest
    
    feature_data, pm2_5s = preprocess(file_name, dest)

    
    data_e_1 = feature_data[0]
    data_e_2 = pm2_5s[0][:, :24, :]
    data_d = pm2_5s[0][:, 24:30, :]

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
    
        input_tensor_1 = torch.from_numpy(data_e_1).to(device)
        input_tensor_2 = torch.from_numpy(data_e_2).to(device)
        target_tensor = torch.from_numpy(data_d).to(device)
        eval_loss, eval_accu, accu_all, counts_p_all, counts_t_all = evaluate(input_tensor_1, input_tensor_2, target_tensor, encoder, decoder)
        print("eval_loss: ", eval_loss)
        print("eval_accu: ", eval_accu)
    else:
        print("train a model first before testing or set path to checkpoint correctly.")