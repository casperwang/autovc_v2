"""generate mel-spectrogram from wav"""
import librosa
from scipy import misc
import pickle
import numpy as np
import audio
import hparams_gen_melspec as hparams
import os
import glob
from tqdm import tqdm

wavs = []
people = dict()
mels = dict()
iters = []
WAV_LEN = 256
PEOPLE_CNT = 20

p = 0
for i in range(225, 225 + PEOPLE_CNT):
	DIR = './VCTK/p'+str(i)
	if os.path.isdir(DIR):
		p += 1
		people[i] = p
		mels[p] = []
		wavs_sz = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
		f, c = 0, 1
		while f < wavs_sz:
			if os.path.isfile(DIR+'/p'+str(i)+'_'+str(c).zfill(3)+'.wav'):
				wavs.append(DIR+'/p'+str(i)+'_'+str(c).zfill(3)+'.wav')
				f += 1
			c += 1

print("finish Checking File!!!")

write_path = './'
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

with open(os.path.join(write_path,'data.pkl'),'wb') as handle:
	pickle.dump(mels, handle)

print("finish 'data.pkl' !!!")

for person in mels.keys():
	for j in range(1, len(mels[person])+1):
		for k in range(1, len(mels[person])+1):
			if j != k:
				iters.append({'i':person, 'j':j, 'k':k})

with open(os.path.join(write_path,'iters.pkl'),'wb') as handle:
	pickle.dump(iters, handle)

print("Finish 'iters.pkl' !!!")
