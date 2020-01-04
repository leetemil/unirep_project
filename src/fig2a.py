# Command-line arguments before anything else
from config import config

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
if NUM_GPUS > 0:
    print(f"Found {NUM_GPUS} GPUs")
    print("CUDNN version:", torch.backends.cudnn.version())
else:
    print("Running on CPU")

# Define model
model = UniRep(config.rnn, config.embed_size, config.hidden_size, config.num_layers)

# Print model information
print(model.summary())

# Use DataParallel if more than 1 GPU
if MULTI_GPU:
    model = nn.DataParallel(model)

model.to(device)

# If using DataParallel, data may be given from CPU (it is automatically distributed to the GPUs)
data_device = torch.device("cuda" if torch.cuda.is_available() and not MULTI_GPU else "cpu")

# Load saved model
saved_dict = torch.load('../data/models/model_LSTM_notrunc.torch', map_location=device)
model_state_dict = saved_dict['model_state_dict']

if any("module" in k for k in model_state_dict.keys()) and not MULTI_GPU:
    # Remove "module" from multi gpu dict
    od = OrderedDict()
    for key, value in model_state_dict.items():
        od[key.replace("module.", "")] = value

    model_state_dict = od

model.load_state_dict(model_state_dict)

# Embed acids using model and show
acids = seq2idx(AMINO_ACIDS, device).unsqueeze(-1)
with torch.no_grad():
    acids = model.embed(acids).squeeze(1).numpy()

pca = PCA(n_components=3)
pca.fit(acids)
pca_acids = pca.transform(acids)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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

xs = pca_acids[:, 0]
ys = pca_acids[:, 1]
zs = pca_acids[:, 2]

ax.scatter(xs, ys, zs, c=colors, s=50)
plt.show()
