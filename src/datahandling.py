from Bio import SeqIO
from torch.utils.data import IterableDataset, DataLoader
import torch

def idxs2seq(idxs):
    return "".join([chr(i + 64) for i in idxs])

class ProteinDataset(IterableDataset):
    def __init__(self, file, device):
        super().__init__()
        self.file = file
        self.device = device

    def __iter__(self):
        def seq2idx(seq):
            return torch.tensor([ord(c) - 64 for c in seq], device = self.device)

        rawSeqs = SeqIO.parse(self.file, "fasta")
        shortSeqs = filter(lambda x: len(x) <= 2000, rawSeqs)
        noBJXZSeqs = filter(lambda x: all(c not in "BJXZ" for c in x), shortSeqs)
        strSeqs = map(lambda x: str(x.seq), noBJXZSeqs)
        encodedSeqs = map(seq2idx, strSeqs)
        return encodedSeqs

def sequenceCollateFn(sequences):
    return torch.nn.utils.rnn.pad_sequence(sequences), [len(s) for s in sequences]

def getProteinDataLoader(proteinDataset, batch_size = 32):
    return DataLoader(proteinDataset, batch_size = batch_size, collate_fn = sequenceCollateFn, pin_memory = True)
