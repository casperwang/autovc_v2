"""Generate mel-spectrogram from wav"""
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
wav_len = 256
mels = []

# read file
wavs.append('')

write_path = './'

for wav_path in tqdm(wavs):

	basename = os.path.basename(wav_path).split('.wav')[0]
	wav = audio.load_wav(wav_path)
	wav = wav / np.abs(wav).max() * hparams.hparams.rescaling_max

	mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
	tmp = mel_spectrogram
	result = np.zeros((wav_len, 80))
	result[:min(tmp.shape[0], wav_len),:tmp.shape[1]] = tmp[:min(tmp.shape[0], wav_len),:tmp.shape[1]]

	mels.append((basename,result))
	misc.imsave(os.path.join(write_path,basename+'.png'), result)
	print(basename, result.shape)

with open(os.path.join(write_path, 'data.pkl'), 'wb') as handle:
	pickle.dump(mels, handle)

