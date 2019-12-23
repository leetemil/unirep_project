# Command-line arguments before anything else
from config import config

from pathlib import Path
import time
import warnings

import torch
from torch import nn

from unirep import UniRep
from datahandling import getProteinDataset, getProteinDataLoader
from constants import *

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

print("Loading data...")
protein_dataset = getProteinDataset(config.data, data_device)
protein_dataloader = getProteinDataLoader(protein_dataset, batch_size = config.batch_size)
print("Data loaded!")

# Define optimizer
opt = torch.optim.Adam(model.parameters())

# Define loss function
criterion = nn.NLLLoss(ignore_index = PADDING_VALUE)

saved_epoch = 0
saved_batch = 0
if config.load_path.is_file():
    loaded = torch.load(config.load_path)
    model.load_state_dict(loaded["model_state_dict"])
    opt.load_state_dict(loaded["optimizer_state_dict"])
    saved_epoch = loaded["epoch"]
    saved_batch = loaded["batch"]
    print("Model loaded succesfully!")

epoch_loss = 0
epoch_loss_count = 0
batch_loss = 0
batch_loss_count = 0
print("Training...")
# Resume epoch count from saved_epoch
for e in range(saved_epoch, config.epochs):
    epoch_start_time = time.time()
    for i, xb in enumerate(protein_dataloader):
        # Run through the data until just after the saved batch
        if saved_batch > 0:
            saved_batch -= 1
            continue

        # Hidden state for new batch should be reset to zero
        last_hidden = None

        for start_idx in range(0, xb.size(1), config.truncation_window):
            # Take optimizer step in the direction of the gradient and reset gradient
            opt.step()
            opt.zero_grad()

            trunc_xb = xb[:, start_idx:start_idx + config.truncation_window]
            mask = (trunc_xb != PADDING_VALUE).to(dtype = torch.long)

            # Forward pass
            pred, last_hidden = model(trunc_xb, last_hidden, mask)

            # Calculate loss
            true = torch.zeros(trunc_xb.shape, dtype = torch.int64, device = device) + PADDING_VALUE
            true[:, :-1] = trunc_xb[:, 1:]

            # Flatten the sequence dimension to compare each timestep in cross entropy loss
            pred = pred.flatten(0, 1)
            true = true.flatten()
            loss = criterion(pred, true)

            epoch_loss += loss.item()
            epoch_loss_count += 1
            batch_loss += loss.item()
            batch_loss_count += 1

            # Calculate gradient which minimizes loss
            loss.backward()

        # Printing progress
        if ((i + 1) % config.print_every) == 0:
            avg_loss = batch_loss / batch_loss_count if batch_loss_count != 0 else -1
            batch_loss = 0
            batch_loss_count = 0
            sequences_processed = (i + 1) * config.batch_size
            time_taken = time.time() - epoch_start_time
            avg_time = time_taken / (i + 1)
            batches_left = (len(protein_dataset) / config.batch_size) - (i + 1)
            eta = max(0, avg_time * batches_left)
            print(f"Epoch: {e:3} Batch: {i + 1:6} Loss: {avg_loss:5.4f} avg. time: {avg_time:5.2f} ETA: {eta / 3600:6.2f} progress: {100 * sequences_processed / len(protein_dataset):6.2f}%")

        # Saving
        if not config.save_path.is_dir() and ((i + 1) % config.save_every) == 0:
            torch.save({
                "epoch": e,
                "batch": i + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss
            }, config.save_path)

    cuda_mem_allocated = 0
    if torch.cuda.is_available:
        cuda_mem_allocated = torch.cuda.max_memory_allocated() / 1024
        torch.cuda.reset_max_memory_allocated()

    avg_loss = epoch_loss / epoch_loss_count if epoch_loss_count != 0 else -1
    epoch_loss = 0
    epoch_loss_count = 0
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    print(f"Epoch {e}: average loss: {avg_loss:5.4f} batches: {i + 1} time: {epoch_time / 3600:.2f} hours. GPU Memory used: {cuda_mem_allocated:.2f} MiB")
