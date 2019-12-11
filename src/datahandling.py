from Bio import SeqIO
from torch.utils.data import IterableDataset, DataLoader
import torch

from constants import *

def idx2seq(idxs):
    return "".join([IDX2SEQ[i] for i in idxs if i != PADDING_VALUE and i != SEQ2IDX[EOS]])

class ProteinDataset(IterableDataset):
    def __init__(self, file, device):
        super().__init__()
        self.file = file
        self.device = device

    def __iter__(self):
        def seq2idx(seq):
            return torch.tensor([SEQ2IDX[s] for s in seq], device = self.device)

        rawSeqs = SeqIO.parse(self.file, "fasta")
        shortSeqs = filter(lambda x: len(x) <= 2000, rawSeqs)
        noBJXZSeqs = filter(lambda x: all(c not in EXCLUDED_AMINO_ACIDS for c in x), shortSeqs)
        strSeqs = map(lambda x: list(str(x.seq)) + [EOS], noBJXZSeqs)
        encodedSeqs = map(seq2idx, strSeqs)
        return encodedSeqs

def sequenceCollateFn(sequences):
    return torch.nn.utils.rnn.pad_sequence(sequences, padding_value = PADDING_VALUE), [len(s) for s in sequences]

def getProteinDataLoader(proteinDataset, batch_size = 32):
    return DataLoader(proteinDataset, batch_size = batch_size, collate_fn = sequenceCollateFn, pin_memory = True)
