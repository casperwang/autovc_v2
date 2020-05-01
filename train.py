import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x, c_org):
        x = x.squeeze(1).transpose(2,1)
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat((x, c_org), dim=1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg):
                
        codes = self.encoder(x, c_org)
        if c_trg is None:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)

'''
import numpy as np
from model import Generator
import torch.autograd as autograd
from torch.autograd import Variable
from resemblyzer import VoiceEncoder, preprocess_wav
from math import ceil
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.functional as F
from params import *
import matplotlib.pyplot as plt
import seaborn as sb
from conversion import convert
#from vocoder import genspec

import pdb
import atexit
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


#Init loss functions
MSELoss = torch.nn.MSELoss()
L1Loss  = torch.nn.L1Loss()

#Tensorboard writer, put to config later
writer = SummaryWriter() #This writes to tensorboard


def pad_seq(x, base = 32):
	len_out = int(base * ceil(float(x.shape[0]) / base))
	len_pad = len_out - x.shape[0]
	assert len_pad >= 0
	return np.pad(x, ((0, len_pad), (0, 0)), "constant"), len_pad

def train_one_epoch(model, optimizer, dataset, device, save_dir, current_iter, current_epoch, doWrite = True): #Takes a PyTorch DataLoader as input and 
	#model: 		the model that you wish to train
	#optimizer:		the optimizer 
	#dataset: 		a PyTorch DataLoader that can be enumerated
	#save_dir: 		directory to save the training weights
	#current_iter: 	what iteration it's currently on (running total)
	#doWrite: 		whether to write to tensorboard or not 
	running_loss = 0
		
	for i, datai in enumerate(tqdm(dataset)):

		#datai: B * C * T * F
		#pdb.set_trace()

		current_iter = current_iter + 1

		uttr_org  = datai["org_uttr"].to(device).double() #
		uttr_trg  = datai["trg_uttr"].to(device).double()  #This and the above will be B * T * F
		emb_org = datai["org_enc"].to(device).double()#
		emb_trg = datai["trg_enc"].to(device).double() #This and the above will be B * 1 * dim_style
		#pdb.set_trace()

		#Turn everything into PyTorch Tensors, and gives the outputs to device


		mel_outputs, mel_outputs_postnet, codes = model(uttr_org, emb_org, emb_trg)
		mel_outputs.squeeze(1)		
		mel_outputs_postnet.squeeze(1)
		#print(torch.norm(mel_outputs_postnet - uttr_trg, 2) / torch.norm(uttr_trg, 2))
		#return
		codes.squeeze(1)

		_, _, trg_codes = model(mel_outputs_postnet, emb_trg, emb_org)
		#mel_outputs: 			the output sans postnet
		#mel_outputs_postnet: 	the above with postnet added
		#codes:					encoder output	
		#pdb.set_trace()

		#Again, get rid of channel dimension
	
		#Zero gradients
		optimizer.zero_grad()
		#Calculate Loss
		L_Recon = MSELoss(mel_outputs_postnet, uttr_org)
		L_Recon0 = MSELoss(mel_outputs, uttr_org)
		L_Content = L1Loss(codes, trg_codes)

		loss = L_Recon * L_Recon + mu * L_Recon0 + lmb * L_Content

		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		print("Loss: " + str(loss.item()))
		print("L_Recon: ")
		print(L_Recon)
		print("L_Content")
		print(L_Content)
		print("Relative Loss: ")
		print(torch.norm((mel_outputs_postnet - uttr_org), 2) / torch.norm(uttr_org))
		if(doWrite == True):
			writer.add_scalar("Loss", loss.item(), current_iter)

		if current_iter % 1000 == 999:
			#Draw trg_uttr
			#Draw mel_outputs_postnet
			#Display loss
			print("Relative Loss: ")
			print(torch.norm((mel_outputs_postnet - uttr_org), 2) / torch.norm(uttr_org))
			torch.save({
				"epoch": current_epoch,
				"model": model.state_dict(),
				"optimizer": optimizer.state_dict()
			}, save_dir + "/test_ckpt_{}iters.ckpt".format(current_iter))
			torch.save({
				"epoch": current_epoch,
				"model": model.state_dict(),
				"optimizer": optimizer.state_dict()
			}, save_dir + "/last.ckpt")
			convert(model, current_iter)
			#genspec("./result_pkl/results_iter{}.pkl".format(current_iter), "res_{}".format(current_iter))



	return current_iters
	'''
