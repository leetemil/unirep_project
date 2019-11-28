import torch
from torch import nn
from torch.nn import functional as F

class Dummy(nn.Module):
	def __init__(self):
		super().__init__()

		self.lin = nn.Linear(42, 42)

	def forward(self, xb):
		return F.relu(self.lin(xb))
