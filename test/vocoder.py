#!/usr/bin/env python
# coding: utf-8

import torch
import librosa
import pickle
from synthesis import build_model
from synthesis import wavegen

spect_vc = pickle.load(open('./results.pkl', 'rb'))
device = torch.device("cpu")
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth", map_location = torch.device(device)) #Using the pretrained WaveNet Vocoder 
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)
    librosa.output.write_wav(name+'.wav', waveform, sr=16000)
