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

os.environ['KMP_DUPLICATE_LIB_OK']='True' #Prevents OMP Error #15

data = datas.testDataset()
metadata = torch.utils.data.DataLoader(data, batch_size=1)

spect_vc = []
def convert(model, current_iter):
	for i, sbmt_i in enumerate(tqdm(metadata)):
		uttr_org = sbmt_i["org_uttr"].to(device).double()
		
		emb_org = sbmt_i['org_enc'].to(device).double()
		
		emb_trg = sbmt_i["trg_enc"].to(device).double()
		
		with torch.no_grad():
			_, x_identic_psnt, _ = model(uttr_org, emb_org, emb_trg)
		
		x_identic_psnt = x_identic_psnt.unsqueeze(0)
		
		uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
		
		spect_vc.append( ('{}x{}'.format(sbmt_i["person"].item(), sbmt_i["person"].item()), uttr_trg) )

	with open('./result_pkl/results_iter{}.pkl'.format(current_iter), 'wb+') as handle:
		pickle.dump(spect_vc, handle)

def convert_test(model, current_iter):
	for i, sbmt_i in enumerate(tqdm(metadata)):
		uttr_org = sbmt_i["org_uttr"].to(device).double()
		
		emb_org = sbmt_i['org_enc'].to(device).double()
		
		emb_trg = sbmt_i["trg_enc"].to(device).double()
		
		with torch.no_grad():
			_, x_identic_psnt, _ = model(uttr_org, emb_org, emb_trg)
		
		x_identic_psnt = x_identic_psnt.unsqueeze(0)
		
		uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
		
		spect_vc.append( ('{}x{}'.format(sbmt_i["p2"].item(), sbmt_i["p1"].item()), uttr_trg) )

	with open('./result_pkl/results_iter{}.pkl'.format(current_iter), 'wb+') as handle:
		pickle.dump(spect_vc, handle)
