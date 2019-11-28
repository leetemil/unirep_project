import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence

class Dummy(nn.Module):
  
  def __init__(self):
    super().__init__()
    # self.lin = nn.Linear(407, 10)
    self.lstm = nn.LSTM(1, 5)
  
  def forward(self, xb):
    packed_xb = pack_sequence(xb)
    out, (hidden, cell) = self.lstm(packed_xb)
    
    return F.relu(out)