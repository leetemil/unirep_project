import torch
from torch import nn
from pathlib import Path
import numpy as np

from datahandling import ProteinDataset, getProteinDataLoader, idx2seq, seq2idx
from constants import *
from unirep import UniRep

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


# Options
MLSTM = False
EMBED_SIZE = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 4

# Get hardware information
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
MULTI_GPU = NUM_GPUS > 1
print(f"Found {NUM_GPUS} GPUs!")
print("CUDNN version:", torch.backends.cudnn.version())

# Define model
model = UniRep(EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, use_mlstm=MLSTM)

# Use DataParallel if more than 1 GPU!
if MULTI_GPU:
    model = nn.DataParallel(model)

model.to(device)

# Apply weight norm on LSTM
if not MLSTM:
    if MULTI_GPU:
        inner_model = model.module
    else:
        inner_model = model

    for i in range(inner_model.num_layers):
        nn.utils.weight_norm(inner_model.rnn, f"weight_ih_l{i}")
        nn.utils.weight_norm(inner_model.rnn, f"weight_hh_l{i}")
        nn.utils.weight_norm(inner_model.rnn, f"bias_ih_l{i}")
        nn.utils.weight_norm(inner_model.rnn, f"bias_hh_l{i}")

# Load saved model
d = torch.load('../data/models/model_LSTM_notrunc.torch', map_location=torch.device('cpu'))
modulestate = d['model_state_dict']

if MULTI_GPU:
  # remove module from multi gpu dict
  from collections import OrderedDict
  od = OrderedDict()
  for key, value in modulestate.items():
    od[key[7:]] = value
  
  modulestate = od

model.load_state_dict(modulestate)

# Embed acids using model and show
acids = seq2idx(AMINO_ACIDS, device).unsqueeze(-1)
acids = model.embed(acids).squeeze(1).detach().numpy()

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

