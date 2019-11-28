from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence

from datahandling import ProteinDataset, getProteinDataLoader

class Dummy(nn.Module):

  def __init__(self):
    super().__init__()
    # self.lin = nn.Linear(407, 10)
    self.lstm = nn.LSTM(1, 5)

  def forward(self, xb):
    out, (hidden, cell) = self.lstm(xb)
    return F.relu(out.data[-1])

# Define model
dummy = Dummy()

# Load data
data_file = Path("../data/uniref50.fasta")
protein_dataset = ProteinDataset(data_file)
protein_dataloader = getProteinDataLoader(protein_dataset, batch_size = 16)

# Define optimizer
opt = torch.optim.Adam(dummy.parameters())

# Define loss function
loss_fn = F.mse_loss

for i, xb in enumerate(protein_dataloader):
    # Forward pass
    pred = dummy(xb)

    # Calculate loss
    loss = loss_fn(pred, torch.ones(5))
    print(loss.item())

    # Calculate gradient which minimizes loss
    loss.backward()

    # Take optimizer step in the direction of the gradient and reset
    opt.step()
    opt.zero_grad()
