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

bottle_neck = 32
dim_style = 256
dim_pre = 512
freq = 32

batch_size = 1

device = 'cpu'
G = Generator(bottle_neck, dim_style, dim_pre, freq).eval().to(device).double()

g_checkpoint = torch.load('autovc.ckpt', map_location = torch.device('cpu')) #AutoVC model weights
G.load_state_dict(g_checkpoint['model'])

data = datas.voiceDataset()
metadata = torch.utils.data.DataLoader([data[0]], batch_size=batch_size)

spect_vc = []

for i, sbmt_i in enumerate(tqdm(metadata)):
    
    uttr_org = sbmt_i["content_uttr"].to(device).double()
    
    emb_org = sbmt_i['content_enc'].to(device).double()
        
    emb_trg = sbmt_i["style_enc"].to(device).double()
    
    with torch.no_grad():
        _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
    
    x_identic_psnt.unsqueeze(0)
    uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
    
    spect_vc.append( ('{}x{}'.format(sbmt_i["person"], sbmt_i["person"]), uttr_trg) )

with open('results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)
