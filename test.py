
import os
import numpy as np
from preprocessing import preprocess
from model import EncoderRNN, DecoderRNN
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(pred, target):
#     set_trace()
    pred = pred.squeeze()
    np_pred = pred.cpu().numpy()
    np_target = target.cpu().numpy()
    d, k = pred.shape
    accu = np.sum(np_pred==np_target)/(d*k)
    return accu

def evaluate(input_tensor, target_tensor, encoder, decoder):
    encoder.eval()
    decoder.eval()
    
    accu_all = []
    
    loss = 0
    criterion = nn.NLLLoss()
    
    with torch.no_grad():
        hn = encoder.initHidden()
        cn = encoder.initHidden()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

    #     encoder_outputs = torch.zeros(input_length, 24, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, (hn, cn) = encoder(
                input_tensor[ei], hn, cn)


        decoder_input = target_tensor[0, :]

        for di in range(target_length):
            decoder_output, (hn, cn) = decoder(decoder_input, hn, cn) 
            loss += criterion(decoder_output, target_tensor[di, :].view(-1,).long())

            decoder_output = decoder_output.view(6, 35 ,6)
            topv, topi = decoder_output.topk(1, dim=2)
            
            accu_all.append(accuracy(topi, target_tensor[di, :]))
            decoder_input = topi.squeeze().detach()
#     set_trace()
    return loss.item() / target_length, np.mean(accu_all)

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="PM2.5 Prediction")
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')
    parser.add_argument('--dest', type=str, default="../air_quality", help='path to dataset.')
    parser.add_argument('--epochs', type=int, default=10, help='epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for training')
    parser.add_argument('--input_type', type=int, default=0, help='0 or 1, choose which input type to run')
    args = parser.parse_args()

    hidden_size = args.hidden_size
    input_size_enc = 35
    input_size_dec = 35
    output_size = 35
    epochs = args.epochs
    learning_rate = args.lr

    file_name = "北京空气质量.zip"
    dest = args.dest
    
    feature_data, pm2_5s_1, pm2_5s_2, pm2_5s_1_2014, pm2_5s_2_2014 = preprocess(file_name, dest)

    checkpoint = '../checkpoint_lstm.pth_'+input_type+'.tar'
    encoder = EncoderRNN(input_size_enc, hidden_size).to(device)
    decoder = DecoderRNN(input_size_dec, hidden_size, output_size).to(device)

    flag = 0
    if flag == 1:
        data_e = feature_data
        data_d = pm2_5s_2[:, :6, :]
    else:
        data_e = pm2_5s_1[:, :24, :]
        data_d = pm2_5s_1[:, 24:24+6, :]

    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint)    
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        
        epoch = checkpoint["epoch"]+1
        encoder_optimizer = checkpoint["encoder_optimizer"]
        decoder_optimizer = checkpoint["decoder_optimizer"]
    
        input_tensor = torch.from_numpy(data_e).to(device)
        target_tensor = torch.from_numpy(data_d).to(device)
        eval_loss, eval_accu = evaluate(input_tensor, target_tensor, encoder, decoder)
        print("eval_loss: ", eval_loss)
        print("eval_accu: ", eval_accu)
    else:
        print("train a model first before testing.")