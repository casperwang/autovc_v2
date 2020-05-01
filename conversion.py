#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#from resemblyzer import preprocess_wav, VoiceEncoder #Style encoder


import os
import pickle
import torch
from tqdm import tqdm
import numpy as np
import data.dataLoader as datas
from math import ceil
from model import Generator
from params import *
import os
import pdb
from resemblyzer import VoiceEncoder

os.environ['KMP_DUPLICATE_LIB_OK']='True' #Prevents OMP Error #15


data = datas.testDataset()
metadata = torch.utils.data.DataLoader([data[0]], batch_size=batch_size)
encoder = VoiceEncoder()

int16_max = (2 ** 15) - 1


def pad_seq(x, base=32):
	len_out = int(base * ceil(float(x.shape[0])/base))
	len_pad = len_out - x.shape[0]
	assert len_pad >= 0
	return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))

def test_same_person(model, name):
	spect_vc = []
	for i, sbmt_i in enumerate(tqdm(metadata)):
		uttr_org = sbmt_i["org_uttr"].to(device).double()
		
		emb_org = sbmt_i['org_enc'].to(device).double()
		
		emb_trg = sbmt_i["org_enc"].to(device).double()
		
		with torch.no_grad():
			_, x_identic_psnt, _ = model(uttr_org, emb_org, emb_trg)
		
		x_identic_psnt = x_identic_psnt.unsqueeze(0)
		
		uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
		
		spect_vc.append( ('{}x{}'.format(sbmt_i["p1"].item(), sbmt_i["p2"].item()), uttr_trg) )

	with open('./result_pkl/{}.pkl'.format(name), 'wb+') as handle:
		pickle.dump(spect_vc, handle)


def convert(model, current_iter):
	spect_vc = []
	for i, sbmt_i in enumerate(tqdm(metadata)):
		uttr_org = sbmt_i["org_uttr"].to(device).double()
		
		emb_org = sbmt_i['org_enc'].to(device).double()
		
		emb_trg = sbmt_i["trg_enc"].to(device).double()
		
		with torch.no_grad():
			_, x_identic_psnt, _ = model(uttr_org, emb_org, emb_trg)
		
		uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
		
		spect_vc.append( ('{}x{}'.format(sbmt_i["p1"].item(), sbmt_i["p2"].item()), uttr_trg) )

	with open('./result_pkl/results_iter{}.pkl'.format(current_iter), 'wb+') as handle:
		pickle.dump(spect_vc, handle)

def convert_two(model, uttr_org, uttr_trg):
	spect_vc = []
	#spect_vc.append( ("uttr_org", uttr_org) )
	#spect_vc.append( ("uttr_trg", uttr_trg) )

	uttr_trg, _ = pad_seq(uttr_trg, 32)
	uttr_org, _ = pad_seq(uttr_org, 32)
	trg_enc = normalize_volume(uttr_trg.reshape(-1), target_dBFS = -30, increase_only = True)
	trg_enc = encoder.embed_utterance(trg_enc)

	org_enc = normalize_volume(uttr_org.reshape(-1), target_dBFS = -30, increase_only = True)
	org_enc = encoder.embed_utterance(trg_enc)

	uttr_trg = torch.FloatTensor(uttr_trg).to(device).double().unsqueeze(0)
	uttr_org = torch.FloatTensor(uttr_org).to(device).double().unsqueeze(0)
	org_enc = torch.FloatTensor(org_enc).to(device).double().unsqueeze(0)
	trg_enc = torch.FloatTensor(trg_enc).to(device).double().unsqueeze(0)

	with torch.no_grad():
			_, x_identic_psnt, _ = model(uttr_org, org_enc, trg_enc)
		
	res = x_identic_psnt[0, 0, :, :].cpu().numpy()
	spect_vc.append( ('fin_conversion', res) )
		
	return spect_vc
