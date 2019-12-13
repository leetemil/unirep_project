from pathlib import Path
import time
import warnings

# Ignore warning about contiguous memory
warnings.filterwarnings("ignore", category = UserWarning)

import torch
from torch import nn
from torch.nn import functional as F

from datahandling import ProteinDataset, getProteinDataLoader, idx2seq
from constants import *

class UniRef(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(NUM_TOKENS, self.embed_size, padding_idx = PADDING_VALUE)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers = self.num_layers, batch_first = True)
        self.lin = nn.Linear(self.hidden_size, NUM_INFERENCE_TOKENS)

    def forward(self, xb, xb_lens):
        longest_length = xb.size(1)
        embedding = self.embed(xb)
        packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(embedding, xb_lens, batch_first = True, enforce_sorted = False)
        packed_out, (hidden, cell) = self.rnn(packed_seqs)
        out, lens = torch.nn.utils.rnn.pad_packed_sequence(packed_out, total_length = longest_length, batch_first = True)
        linear = self.lin(out)
        return linear

# Get hardware
print("CUDNN version:", torch.backends.cudnn.version())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
MULTI_GPU = NUM_GPUS > 1
print(f"Found {NUM_GPUS} GPUs!")

# Define model
EMBED_SIZE = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 4

model = UniRef(EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)

# This currently does not play nicely with data parallel
# Apply weight norm on LSTM
# for i in range(model.num_layers):
#     nn.utils.weight_norm(model.rnn, f"weight_ih_l{i}")
#     nn.utils.weight_norm(model.rnn, f"weight_hh_l{i}")
#     nn.utils.weight_norm(model.rnn, f"bias_ih_l{i}")
#     nn.utils.weight_norm(model.rnn, f"bias_hh_l{i}")

# Use DataParallel if more than 1 GPU!
if MULTI_GPU:
    model = nn.DataParallel(model)

model.to(device)

EPOCHS = 1000
BATCH_SIZE = 4
PRINT_EVERY = 100
SAVE_EVERY = 100

# Load data
# data_file = Path("../data/dummy/uniref-id_UniRef50_A0A007ORid_UniRef50_A0A009DWD5ORid_UniRef50_A0A009D-.fasta")
data_file = Path("../data/UniRef50/uniref50.fasta")
data_device = torch.device("cuda" if torch.cuda.is_available() and not MULTI_GPU else "cpu")
protein_dataset = ProteinDataset(data_file, data_device)
protein_dataloader = getProteinDataLoader(protein_dataset, batch_size = BATCH_SIZE)

# Define optimizer
opt = torch.optim.Adam(model.parameters())

# Define loss function
loss_fn = nn.CrossEntropyLoss(ignore_index = PADDING_VALUE)

for e in range(EPOCHS):
    total_time = 0
    for i, (xb, xb_lens) in enumerate(protein_dataloader):
        start_time = time.time()
        # Forward pass
        pred = model(xb, xb_lens)

        # Calculate loss
        true = torch.zeros(xb.shape, dtype = torch.int64, device = device) + PADDING_VALUE
        true[:, :-1] = xb[:, 1:]

        pred = pred.flatten(0, 1)
        true = true.flatten()
        loss = loss_fn(pred, true)

        # Calculate gradient which minimizes loss
        loss.backward()

        # Take optimizer step in the direction of the gradient and reset
        opt.step()
        opt.zero_grad()
        end_time = time.time()
        loop_time = end_time - start_time
        total_time += loop_time
        if (i % PRINT_EVERY) == 0:
            print(f"Epoch: {e:6} Batch: {i:6} Loss: {loss.item():5.4f} time: {loop_time:5.2f}, avg. time: {total_time / (i + 1):5.2f} progress: {100 * i * BATCH_SIZE / 36000000:7.3f}%")

        if (i % SAVE_EVERY) == 0:
            torch.save({
                "iteration": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss
            }, "model.torch")
