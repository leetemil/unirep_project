from Bio import SeqIO
from torch.utils.data import IterableDataset, DataLoader
import torch

def seq2idx(seq):
    return torch.Tensor([ord(c) - 65 for c in seq]).unsqueeze(-1)

def idxs2seq(idxs):
    return "".join([chr(i + 65) for i in idxs])

class ProteinDataset(IterableDataset):
    def __init__(self, file):
        super().__init__()
        self.file = file

    def __iter__(self):
        rawSeqs = SeqIO.parse(self.file, "fasta")
        strSeqs = map(lambda x: str(x.seq), rawSeqs)
        encodedSeqs = map(seq2idx, strSeqs)
        return encodedSeqs

def sequenceCollateFn(sequences):
    return torch.nn.utils.rnn.pad_sequence(sequences)

def getProteinDataLoader(proteinDataset, batch_size = 32):
    return DataLoader(proteinDataset, batch_size = batch_size, collate_fn = sequenceCollateFn)
