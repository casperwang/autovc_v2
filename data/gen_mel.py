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
PEOPLE_CNT = 20
write_path = './'
int16_max = (2 ** 15) - 1
encoder = VoiceEncoder()

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

p = 0
for i in range(225, 225 + PEOPLE_CNT):
	DIR = './VCTK/p'+str(i)
	if os.path.isdir(DIR):
		p += 1
		people[i] = p
		mels[p] = []
		style_list[p] = []
		style[p] =[0]*256
		wavs_sz = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
		f, c = 0, 1
		while f < wavs_sz:
			if os.path.isfile(DIR+'/p'+str(i)+'_'+str(c).zfill(3)+'.wav'):
				wavs.append(DIR+'/p'+str(i)+'_'+str(c).zfill(3)+'.wav')
				f += 1
			c += 1

print("finish Checking File!!!")
for wav_path in tqdm(wavs):

	basename = os.path.basename(wav_path).split('.wav')[0]
	idx = people[int(basename[-7:-4])]
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

for idx in style_list:
	for s in style_list[idx]:
		for i in range(256):
			style[idx][i] += s[i]
	for i in range(256):
		style[idx][i] = style[idx][i] / len(style_list[idx])

with open(os.path.join(write_path,'style_data.pkl'),'wb') as handle:
	pickle.dump(style, handle)

print("finish 'style_data.pkl' !!!")

'''
for person in mels.keys():
	for j in range(0, len(mels[person])):
		for k in range(0, len(mels[person])):
			if j != k:
				iters.append({'i':person, 'j':j, 'k':k})
with open(os.path.join(write_path,'test_iters.pkl'),'wb') as handle:
	pickle.dump(iters, handle)
print("Finish 'iters.pkl' !!!")
'''

iters.append({'p1':1, 'p2':1, 'i':0, 'j':0})
iters.append({'p1':2, 'p2':2, 'i':0, 'j':0})
iters.append({'p1':1, 'p2':2, 'i':0, 'j':0})
iters.append({'p1':2, 'p2':1, 'i':0, 'j':0})

with open(os.path.join(write_path,'test_iters.pkl'),'wb') as handle:
	pickle.dump(iters, handle)

print("Finish 'test_iters.pkl' !!!")
