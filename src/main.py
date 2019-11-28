from datahandling import ProteinDataset
from dummy import Dummy
from pathlib import Path
import numpy as np
import torch

filepath = Path('../data/dummydata.fasta')

# data = torch.Tensor(list(create_dataset(filepath))[0])

data = ProteinDataset(filepath)
loader = torch.utils.data.DataLoader(data, batch_size=10)

model = Dummy()


# print(model(loader))