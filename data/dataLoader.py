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
	style_folder = []

	def __init__(self):
		self.iter_folder = pickle.load(open('./data/iters.pkl', "rb"))
		self.style_folder = pickle.load(open('./data/style_data.pkl', "rb"))
		self.wav_folder = pickle.load(open('./data/data.pkl', "rb"))
	
	def __getitem__(self, index): #Should iterate through all possible triples
		item = dict()
		p1 = self.iter_folder[index]['i']
		
		trg_uttr, _ = pad_seq(self.wav_folder[p1][self.iter_folder[index]['j']], 32)
		org_uttr, _ = pad_seq(self.wav_folder[p1][self.iter_folder[index]['k']], 32)
		
		trg_enc = style_folder[p1]
		org_enc = style_folder[p1]
		
		item["person"] = p1
		item["trg_uttr"] = trg_uttr
		item["org_uttr"] = org_uttr
		item["trg_enc"] = trg_enc
		item["org_enc"] = org_enc

		return item
	
	def __len__(self):
		return len(self.iter_folder)

class testDataset(Dataset):
	wav_folder = []
	iter_folder = []
	style_folder = []

	def __init__(self):
		self.iter_folder = pickle.load(open('./data/test_iters.pkl', "rb"))
		self.style_folder = pickle.load(open('./data/style_data.pkl', "rb"))
		self.wav_folder = pickle.load(open('./data/data.pkl', "rb"))
	
	def __getitem__(self, index): #Should iterate through all possible triples
		item = dict()
		p1 = self.iter_folder[index]['p1']
		p2 = self.iter_folder[index]['p2']
		
		trg_uttr, _ = pad_seq(self.wav_folder[p1][self.iter_folder[index]['i']], 32)
		org_uttr, _ = pad_seq(self.wav_folder[p2][self.iter_folder[index]['j']], 32)
		
		trg_enc = style_folder[p1]
		org_enc = style_folder[p2]

		item["p1"] = p1
		item["p2"] = p2
		item["trg_uttr"] = trg_uttr
		item["org_uttr"] = org_uttr
		item["trg_enc"] = trg_enc
		item["org_enc"] = org_enc

		return item
	
	def __len__(self):
		return len(self.iter_folder)