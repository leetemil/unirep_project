from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence, pad_sequence

from datahandling import ProteinDataset, getProteinDataLoader
from mlstm import script_mlstm, LSTMState

class Dummy(nn.Module):

    def __init__(self):
        super().__init__()
        # self.lin = nn.Linear(407, 10)
        self.input_size = 1
        self.hidden_size = 5
        self.mlstm = script_mlstm(self.input_size, self.hidden_size)

    def forward(self, xb, state):
        out, state = self.mlstm(xb, state)
        return F.relu(out[-1, :, :])

    def init_hidden(self, batch_size):
        # List of 1 state, since it has a state per layer (1 layer here)
        return [LSTMState(torch.zeros(batch_size, self.hidden_size), torch.zeros(batch_size, self.hidden_size))]

# Define model
dummy = Dummy()

# Load data
data_file = Path("../data/UniRef50/uniref50.fasta")
protein_dataset = ProteinDataset(data_file)
protein_dataloader = getProteinDataLoader(protein_dataset, batch_size = 16)

# Define optimizer
opt = torch.optim.Adam(dummy.parameters())

# Define loss function
loss_fn = F.mse_loss

for i, xb in enumerate(protein_dataloader):
    # if i >= 99:
    #     print(xb)
    #     print(xb.shape)
    # print(f"Iteration: {i}")
    # Forward pass
    padded_seq_len, batch_size, features = xb.shape
    init_hidden = dummy.init_hidden(batch_size)
    pred = dummy(xb, init_hidden)

    # Calculate loss
    loss = loss_fn(pred, torch.ones(16, 5))
    print(loss.item())

    # Calculate gradient which minimizes loss
    loss.backward()

    # Take optimizer step in the direction of the gradient and reset
    opt.step()
    opt.zero_grad()
    # print(f"Iteration: {i} end")
