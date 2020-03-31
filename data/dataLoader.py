import pickle
from sklearn import preprocessing
import numpy as np
import torch
from math import ceil
from torch.utils.data import Dataset, DataLoader

from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pdb


int16_max = (2 ** 15) - 1
encoder = VoiceEncoder()

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

class voiceDataset(Dataset):
	wav_folder = []
	iter_folder = []

	def __init__(self):
		self.iter_folder = pickle.load(open('./data_loader/iters.pkl', "rb"))
		self.wav_folder = pickle.load(open('./data_loader/data.pkl', "rb"))
	
	def __getitem__(self, index): #Should iterate through all possible triples
		item = dict()
		idx = self.iter_folder[index]["i"]
		
		style_uttr, _ = pad_seq(self.wav_folder[idx][self.iter_folder[index]["j"]], 32)
		content_uttr, _ = pad_seq(self.wav_folder[idx][self.iter_folder[index]["k"]], 32)
		
		st_shape = style_uttr.shape
		cont_shape = content_uttr.shape

		style_uttr = normalize_volume(style_uttr.reshape(-1), target_dBFS = -30, increase_only = True)
		content_uttr = normalize_volume(content_uttr.reshape(-1), target_dBFS = -30, increase_only = True)

		style_enc = encoder.embed_utterance(style_uttr)
		content_enc = encoder.embed_utterance(content_uttr)

		style_uttr = style_uttr.reshape(st_shape)
		content_uttr = content_uttr.reshape(cont_shape)

		item["person"] = idx
		item["style_uttr"] = style_uttr
		item["content_uttr"] = content_uttr
		item["style_enc"] = style_enc
		item["content_enc"] = content_enc

		return item
	
	def __len__(self):
		return len(self.iter_folder)

