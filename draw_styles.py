#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:57:05 2020

@author: liusean
"""

from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import numpy as np
import seaborn as sns 
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from resemblyzer import sampling_rate
from matplotlib import cm
from time import sleep, perf_counter as timer
from umap import UMAP
from sys import stderr
import matplotlib.pyplot as plt
import numpy as np

_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_my_colors = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255 

sns.reset_orig()  # get default matplotlib styles back
clrs = sns.color_palette('husl', n_colors=20)  # a list of RGB tuples0

def plot_projections(embeds, speakers, ax=None, colors=None, markers=None, legend=True, 
                     title="", **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        
    # Compute the 2D projections. You could also project to another number of dimensions (e.g. 
    # for a 3D plot) or use a different different dimensionality reduction like PCA or TSNE.
    reducer = UMAP(**kwargs)
    projs = reducer.fit_transform(embeds)
    
    # Draw the projections
    speakers = np.array(speakers)
    for i, speaker in enumerate(np.unique(speakers)):
        speaker_projs = projs[speakers == speaker]
        marker = "o" if markers is None else markers[i]
        label = speaker if legend else None
        ax.scatter(*speaker_projs.T, marker=marker, c = clrs[:len(speakers)], label=label)
        
    if legend:
        ax.legend(bbox_to_anchor=(1.1, 1.05), title = "Speakers", ncol = 2, fancybox=True, shadow=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    
    return projs


encoder = VoiceEncoder()

allencs = []
speakers = []

plt.figure(figsize=(10, 10), dpi=100)
for i in tqdm(range(225, 247)):
    PATH = "./data/test_vctk/p" + str(i)
    if(os.path.isdir(PATH)):
        for j in range(1, 30):
            wav_dir = PATH + '/p' + str(i) + "_" + str(j).zfill(3) + ".wav"
            if(os.path.isfile(wav_dir)):
                wav = preprocess_wav(wav_dir)
                allencs.append(encoder.embed_utterance(wav))
                speakers.append("p" + str(i))

plot_projections(allencs, speakers, title="Embedding projections")
plt.show()
plt.savefig('styles_scatter.png')