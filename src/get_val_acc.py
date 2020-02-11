import time
from pathlib import Path

import torch
import torch.nn as nn
from unirep import UniRep
from datahandling import getProteinDataset, getProteinDataLoader
from constants import *

device = torch.device("cuda")

sd = torch.load("mLSTM_512.best", map_location = device)["model_state_dict"]

model = UniRep("mLSTM", 10, 512, 1)
model.to(device)
model.load_state_dict({k.replace("module.", ""): sd[k] for k in sd.keys()})

# Define loss function
criterion = nn.NLLLoss(ignore_index = PADDING_VALUE)

model.eval()
torch.set_grad_enabled(False)

validation_protein_dataset = getProteinDataset(Path("../data/preprocessed/proteins10240.txt"), device)
validation_protein_dataloader = getProteinDataLoader(validation_protein_dataset, batch_size = 256)

# Get validation loss and accuracy
val_loss = 0
val_loss_count = 0
correct = 0
count = 0
for i, xb in enumerate(validation_protein_dataloader):
	start_time = time.time()
	print(f"Batch: {i + 1}")
	mask = (xb != PADDING_VALUE).to(dtype = torch.long)

	print(f"Forward ({time.time() - start_time})")
	# Forward pass
	pred, _ = model(xb, None, mask)

	print(f"Loss ({time.time() - start_time})")
	# Calculate loss
	true = torch.zeros(xb.shape, dtype = torch.int64, device = device) + PADDING_VALUE
	true[:, :-1] = xb[:, 1:]

	# Flatten the sequence dimension to compare each timestep in cross entropy loss
	pred = pred.flatten(0, 1)
	true = true.flatten()
	loss = criterion(pred, true)
	val_loss += loss.item()
	val_loss_count += 1
	mask = mask.flatten().to(bool)
	pred = pred[mask]
	true = true[mask]
	count += pred.size(0)
	correct += (pred.argmax(dim = 1) == true).sum().item()
	print(f"Done ({time.time() - start_time}) with accuracy {correct / count}")
print(f"Batches: {i + 1}")
accuracy = correct / count
avg_val_loss = val_loss / val_loss_count

print(f"Validation loss: {avg_val_loss:5.3f}, accuracy: {accuracy:5.3f}")
