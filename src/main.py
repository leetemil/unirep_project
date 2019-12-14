from pathlib import Path
import time
import warnings

# Ignore warning about contiguous memory
warnings.filterwarnings("ignore", category = UserWarning)

import torch
from torch import nn

from unirep import UniRep
from datahandling import ProteinDataset, getProteinDataLoader
from constants import *

# Get hardware information
print("CUDNN version:", torch.backends.cudnn.version())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
MULTI_GPU = NUM_GPUS > 1
print(f"Found {NUM_GPUS} GPUs!")

# Define model
EMBED_SIZE = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 4
model = UniRep(EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)

# Use DataParallel if more than 1 GPU!
if MULTI_GPU:
    model = nn.DataParallel(model)

model.to(device)

if MULTI_GPU:
    inner_model = model.module
else:
    inner_model = model

# Apply weight norm on LSTM
for i in range(inner_model.num_layers):
    nn.utils.weight_norm(inner_model.rnn, f"weight_ih_l{i}")
    nn.utils.weight_norm(inner_model.rnn, f"weight_hh_l{i}")
    nn.utils.weight_norm(inner_model.rnn, f"bias_ih_l{i}")
    nn.utils.weight_norm(inner_model.rnn, f"bias_hh_l{i}")

# Training parameters
EPOCHS = 1000
BATCH_SIZE = 1024
NUM_BATCHES = 1 + (NUM_SEQUENCES // BATCH_SIZE)
PRINT_EVERY = 1000
SAVE_EVERY = 1000

# Load data
# data_file = Path("../data/dummy/uniref-id_UniRef50_A0A007ORid_UniRef50_A0A009DWD5ORid_UniRef50_A0A009D-.fasta")
data_file = Path("../data/UniRef50/uniref50.fasta")

# If using DataParallel, data may be given from CPU (it is automatically distributed to the GPUs)
data_device = torch.device("cuda" if torch.cuda.is_available() and not MULTI_GPU else "cpu")
protein_dataset = ProteinDataset(data_file, data_device)
protein_dataloader = getProteinDataLoader(protein_dataset, batch_size = BATCH_SIZE)

# Define optimizer
opt = torch.optim.Adam(model.parameters())

# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index = PADDING_VALUE)

# Train
for e in range(EPOCHS):
    epoch_start_time = time.time()
    for i, (xb, xb_lens) in enumerate(protein_dataloader):
        # Forward pass
        pred = model(xb, xb_lens)

        # Calculate loss
        true = torch.zeros(xb.shape, dtype = torch.int64, device = device) + PADDING_VALUE
        true[:, :-1] = xb[:, 1:]

        # Flatten the sequence dimension to compare each timestep in cross entropy loss
        pred = pred.flatten(0, 1)
        true = true.flatten()
        loss = criterion(pred, true)

        # Calculate gradient which minimizes loss
        loss.backward()

        # Printing progress
        sequences_processed = (i + 1) * BATCH_SIZE
        time_taken = time.time() - epoch_start_time
        avg_time = (time_taken) / (i + 1)
        batches_left = NUM_BATCHES - (i + 1)
        eta = max(0, avg_time * batches_left)
        if (i % PRINT_EVERY) == 0:
            print(f"Epoch: {e:3} Batch: {i:6} Loss: {loss.item():5.4f} avg. time: {avg_time:5.2f} ETA: {eta / 3600:5.2f} progress: {100 * sequences_processed / NUM_SEQUENCES:6.2f}%")

        # Saving
        if (i % SAVE_EVERY) == 0:
            torch.save({
                "iteration": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss
            }, "model.torch")

        # Take optimizer step in the direction of the gradient and reset gradient
        # Should be done after saving, since we otherwise save a different model than the one we reported on
        opt.step()
        opt.zero_grad()

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    print(f"Epoch {e} with {i + 1} batches took {epoch_time / 3600:.2f} hours ({(epoch_time / (i + 1)) / 3600:.2f} per batch).")
