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

		#Turn everything into PyTorch Tensors, and gives the outputs to device

		mel_outputs, mel_outputs_postnet, codes = model(uttr_org, emb_org, emb_trg)
		mel_outputs = mel_outputs.squeeze(1)		
		mel_outputs_postnet = mel_outputs_postnet.squeeze(1)
		#print(torch.norm(mel_outputs_postnet - uttr_trg, 2) / torch.norm(uttr_trg, 2))
		#return

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
		'''print("Loss: " + str(loss.item()))
		print("L_Recon: ")
		print(L_Recon)
		print("L_Content")
		print(L_Content)
		print("Relative Loss: ")
		print(torch.norm((mel_outputs_postnet - uttr_org), 2) / torch.norm(uttr_org))'''
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

