# Command-line arguments before anything else
from collections import OrderedDict

import torch
from torch import nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from datahandling import ProteinDataset, IterProteinDataset, getProteinDataLoader, idx2seq, seq2idx
from constants import *
from unirep import UniRep

# Get hardware information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
MULTI_GPU = NUM_GPUS > 1

MODEL_TYPE = 'GRU'

if NUM_GPUS > 0:
    print(f"Found {NUM_GPUS} GPUs")
    print("CUDNN version:", torch.backends.cudnn.version())
else:
    print("Running on CPU")

# Define model
model = UniRep(MODEL_TYPE, 10, 1024, 1)

# Print model information
print(model.summary())

# Use DataParallel if more than 1 GPU
if MULTI_GPU:
    model = nn.DataParallel(model)

model.to(device)

# If using DataParallel, data may be given from CPU (it is automatically distributed to the GPUs)
data_device = torch.device("cuda" if torch.cuda.is_available() and not MULTI_GPU else "cpu")

# Load saved model
saved_dict = torch.load(f'../data/models/{MODEL_TYPE}_new_vocab.best', map_location=device)
model_state_dict = saved_dict['model_state_dict']

if any("module" in k for k in model_state_dict.keys()) and not MULTI_GPU:
    # Remove "module" from multi gpu dict
    od = OrderedDict()
    for key, value in model_state_dict.items():
        od[key.replace("module.", "")] = value

    model_state_dict = od

model.load_state_dict(model_state_dict)

AMINO_ACIDS = ["M", "R", "H", "K", "D", "E", "S", "T", "N", "Q", "C", "U", "G", "P", "A", "V", "I", "F", "Y", "W", "L", "O"]

# Embed acids using model and show
acids = seq2idx(AMINO_ACIDS, device).unsqueeze(-1)
with torch.no_grad():
    acids = model.embed(acids).squeeze(1).numpy()

pca = PCA(n_components=3)
pca.fit(acids)
pca_acids = pca.transform(acids)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'{MODEL_TYPE}: Amino Acid Embedding')

AILMV = 'aqua'
GP = 'darkblue'
CNQST = 'darkgreen'
DE = 'gold'
FWY = 'purple'
HKR = 'salmon'

colors = [
    AILMV, # A
    CNQST, # C
    DE, # D
    DE, # E
    FWY, # F
    GP, # G
    HKR, # H
    AILMV, # I
    HKR, # K
    AILMV, # L
    AILMV, # M
    CNQST, # N
    'white', # O
    GP, # P
    CNQST, # Q
    HKR, # R
    CNQST, # S
    CNQST, # T
    'white',# U
    AILMV, # V
    FWY, # W
    FWY, # Y
]

import matplotlib.patches as mpatches
uniq_colors = ['aqua', 'darkblue', 'darkgreen', 'gold', 'purple', 'salmon']
labels = ['Hydrophobic aliphatic', 'Unique', 'Polar Neutral', 'Charged acidic', 'Hydrophobic aromatic', 'Charged basic']
patches = [mpatches.Patch(color=c, label=l) for (c, l) in zip(uniq_colors, labels)]

xs = pca_acids[:, 0]
ys = pca_acids[:, 1]
zs = pca_acids[:, 2]

ax.scatter(xs, ys, zs, c=colors, s=50)
ax.legend(handles=patches, loc='lower left')
ax.view_init(9, -53)
plt.tight_layout()
plt.savefig(f'../figures/fig2a_{MODEL_TYPE}.pdf')
plt.show()
