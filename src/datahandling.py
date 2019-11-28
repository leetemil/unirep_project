from Bio import SeqIO
from torch.utils.data import IterableDataset
import torch

def seq2idx(seq):
  return [ord(c) - 65 for c in seq]

def idxs2seq(idxs):
  return "".join([chr(i + 65) for i in idxs])

def fasta_loader(filepath):
  raw = SeqIO.parse(filepath, "fasta")
  data = map(lambda x: str(x.seq), raw)
  
  return data
  
def create_dataset(filepath):
  data = fasta_loader(filepath)
  encoded = map(seq2idx, data)
  
  return encoded
  
class ProteinDataset(IterableDataset):
  def __init__(self, file):
    super().__init__()
    self.file = file
    self.data = create_dataset(file)
  
  def __iter__(self):
    return iter(self.data)