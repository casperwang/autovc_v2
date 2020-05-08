"""generate mel-spectrogram from wav"""
import librosa
from scipy import misc
import pickle
import numpy as np
from math import ceil
import audio
import hparams_gen_melspec as hparams
from resemblyzer import preprocess_wav, VoiceEncoder
import os
import glob
from tqdm import tqdm

wavs = []
people = dict()
mels = dict()
style_list = dict()
style = dict()
iters = []
WAV_LEN = 256
write_path = './'
int16_max = (2 ** 15) - 1
encoder = VoiceEncoder()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def pad_seq(x, base=32):
	len_out = int(base * ceil(float(x.shape[0])/base))
	len_pad = len_out - x.shape[0]
	assert len_pad >= 0
	return np.pad(x, ((0,len_pad),(0,0)), 'constant')

def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))

org_DIR = 'data/Sean_01.wav'
trg_DIR = 'data/C3PO_01.wav'

if os.path.isfile(org_DIR):
    mels[1] = []
    style_list[1] = []
    style[1] = [0]*256
    people[org_DIR[5:-7]] = 1
    wavs.append(org_DIR)

if os.path.isfile(trg_DIR):
    mels[2] = []
    style_list[2] = []
    style[2] = [0]*256
    people[trg_DIR[5:-7]] = 2
    wavs.append(trg_DIR)

print("finish Checking File!!!")

for wav_path in tqdm(wavs):
	basename = os.path.basename(wav_path).split('.wav')[0]
	idx = people[basename[0:-3]]
	wav = audio.load_wav(wav_path)
	wav = wav / np.abs(wav).max() * hparams.hparams.rescaling_max

	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T

	tmp = mel_spectrogram
	result = np.zeros((WAV_LEN, 80))
	result[:min(tmp.shape[0],WAV_LEN),:tmp.shape[1]] = tmp[:min(tmp.shape[0],WAV_LEN),:tmp.shape[1]]

	mels[idx].append(result)

	result = normalize_volume(result.reshape(-1), target_dBFS = -30, increase_only = True)
	result = encoder.embed_utterance(result)
	style_list[idx].append(result)

with open(os.path.join(write_path,'data.pkl'),'wb') as handle:
	pickle.dump(mels, handle)

print("finish 'data.pkl' !!!")
