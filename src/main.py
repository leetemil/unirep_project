from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from datahandling import ProteinDataset, getProteinDataLoader

class UniRef(nn.Module):

    def __init__(self):
        super().__init__()
        # self.lin = nn.Linear(407, 10)
        self.input_size = 10
        self.hidden_size = 64
        self.num_layers = 1

        self.embed = nn.Embedding(26, 10, padding_idx = 0)
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers = self.num_layers)
        self.lin = nn.Linear(64, 26)

    def forward(self, xb, xb_lens):
        seq_len, batch_size = xb.shape
        embedding = self.embed(xb)
        packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(embedding, xb_lens, enforce_sorted = False)
        out, (hidden, cell) = self.rnn(packed_seqs)
        padded_out, lens = torch.nn.utils.rnn.pad_packed_sequence(out)
        linear = self.lin(padded_out)
        return linear

# Define model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dummy = Dummy().to(device)

# Load data
data_file = Path("../data/UniRef50/uniref50.fasta")
protein_dataset = ProteinDataset(data_file, device)
protein_dataloader = getProteinDataLoader(protein_dataset, batch_size = 32)

# Define optimizer
opt = torch.optim.Adam(dummy.parameters())

# Define loss function
loss_fn = nn.CrossEntropyLoss(ignore_index = 0)

train_iters = 100
total_time = 0
for i, (xb, xb_lens) in enumerate(protein_dataloader):
    start_time = time.time()
    # Forward pass
    pred = dummy(xb, xb_lens)

    # Calculate loss
    true = torch.zeros(xb.shape, dtype = torch.int64)
    true[:-1, :] = xb[1:, :]

    # Permute to correct shape for loss
    pred = pred.permute(1, 2, 0)
    true = true.permute(1, 0)
    loss = loss_fn(pred, true)

    # Calculate gradient which minimizes loss
    loss.backward()

    # Take optimizer step in the direction of the gradient and reset
    opt.step()
    opt.zero_grad()
    end_time = time.time()
    loop_time = end_time - start_time
    total_time += loop_time
    print(f"Iteration: {i:4}, loss: {loss.item():5.4f} time: {loop_time:5.2f}, avg. time: {total_time / (i + 1):5.2f}, padded length: {xb.size(0):4}")
    if i > train_iters:
        break
