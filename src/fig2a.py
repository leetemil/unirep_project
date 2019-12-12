import torch
from torch import nn
from pathlib import Path

from datahandling import ProteinDataset, getProteinDataLoader, idx2seq, seq2idx
from constants import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


class UniRef(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(NUM_TOKENS, self.embed_size, padding_idx = PADDING_VALUE)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers = self.num_layers)
        self.lin = nn.Linear(self.hidden_size, NUM_INFERENCE_TOKENS)

    def forward(self, xb, xb_lens):
        embedding = self.embed(xb)
        packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(embedding, xb_lens, enforce_sorted = False)
        out, (hidden, cell) = self.rnn(packed_seqs)
        padded_out, lens = torch.nn.utils.rnn.pad_packed_sequence(out)
        linear = self.lin(padded_out)
        return linear

# Define model
EMBED_SIZE = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 4
model = UniRef(EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Apply weight norm on LSTM
for i in range(model.num_layers):
    nn.utils.weight_norm(model.rnn, f"weight_ih_l{i}")
    nn.utils.weight_norm(model.rnn, f"weight_hh_l{i}")
    nn.utils.weight_norm(model.rnn, f"bias_ih_l{i}")
    nn.utils.weight_norm(model.rnn, f"bias_hh_l{i}")

# Load saved model
d = torch.load('model.torch', map_location=torch.device('cpu'))
model.load_state_dict(d['model_state_dict'])

# Embed acids using model and show
acids = seq2idx(AMINO_ACIDS, device).unsqueeze(-1)
acids = model.embed(acids).squeeze(1).detach().numpy()

pca = PCA(n_components=3)
pca.fit(acids)

pca_acids = pca.transform(acids)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = pca_acids[:, 0]
ys = pca_acids[:, 1]
zs = pca_acids[:, 2]

ax.scatter(xs, ys, zs)
plt.show()

