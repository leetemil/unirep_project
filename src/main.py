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

        # Define parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.embed = nn.Embedding(NUM_TOKENS, self.embed_size, padding_idx = PADDING_VALUE)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers = self.num_layers, batch_first = True)
        self.lin = nn.Linear(self.hidden_size, NUM_INFERENCE_TOKENS)

    def forward(self, xb, xb_lens):
        # This is important to do for DataParallel. We need this length when padding the packed sequence again
        longest_length = xb.size(1)

        # Convert indices to embedded vectors
        embedding = self.embed(xb)

        # Convert padded batch to a packed sequence. This makes it so the LSTM does not compute anything for the pad values
        packed_seqs = torch.nn.utils.rnn.pack_padded_sequence(embedding, xb_lens, batch_first = True, enforce_sorted = False)

        # Output from RNN is also packed when given packed
        packed_out, (hidden, cell) = self.rnn(packed_seqs)

        # Convert packed sequence back into padded batch. Using the longest length from the original batch is important
        # as when using DataParallel, the padded batch may be longer than the longest sequence in the batch,
        # since the longest sequence may have been given to another GPU. Doing it this way ensures that all GPUs produce
        # an output batch that is the same dimensions.
        out, lens = torch.nn.utils.rnn.pad_packed_sequence(packed_out, total_length = longest_length, batch_first = True)

        # Linear layer to convert from RNN hidden size -> inference tokens scores
        linear = self.lin(out)
        return linear

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

model = UniRef(EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS)

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

EPOCHS = 1000
BATCH_SIZE = 1024
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

        # Take optimizer step in the direction of the gradient and reset gradient
        opt.step()
        opt.zero_grad()

        # Printing progress
        avg_time = (time.time() - epoch_start_time) / (i + 1)
        if (i % PRINT_EVERY) == 0:
            print(f"Epoch: {e:3} Batch: {i:6} Loss: {loss.item():5.4f} avg. time: {avg_time:5.2f} progress: {100 * (i + 1) * BATCH_SIZE / NUM_SEQUENCES:6.2f}%")

        # Saving
        if (i % SAVE_EVERY) == 0:
            torch.save({
                "iteration": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "loss": loss
            }, "model.torch")
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    print(f"Epoch {e} with {i + 1} batches took {epoch_time / 3600:.2f} hours ({(epoch_time / (i + 1)) / 3600:.2f} per batch).")
