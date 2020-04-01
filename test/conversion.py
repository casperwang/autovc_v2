#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#from resemblyzer import preprocess_wav, VoiceEncoder #Style encoder


import os
import pickle
import torch
import numpy as np
import autovc-v2.data.dataLoader as datas
from math import ceil
from model import Generator

device = 'cpu'
G = Generator(32,256,512,32).eval().to(device)

g_checkpoint = torch.load('../autovc.ckpt', map_location = torch.device('cpu')) #AutoVC model weights
G.load_state_dict(g_checkpoint['model'])

data = datas.voiceDataset()
metadata = [data[0]]

spect_vc = []

for sbmt_i in metadata:
    
    x_org = sbmt_i['spectrogram']
    
    uttr_org = x_org
    
    emb_org = sbmt_i['style'][np.newaxis, :]
    
    for sbmt_j in metadata:
        
        emb_trg = sbmt_j["style"][np.newaxis, :]

        tmp = np.zeros((256), dtype='float64')
        tmp[0] = 1
        
        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, torch.from_numpy(tmp).cpu().float()[np.newaxis, :], emb_trg)
        
        uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        
        spect_vc.append( ('{}x{}'.format(sbmt_i["person"], sbmt_j["person"]), uttr_trg) )

with open('results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)
