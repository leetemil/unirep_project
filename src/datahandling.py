import random

from Bio import SeqIO
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch

from constants import *

def seq2idx(seq, device = None):
    return torch.tensor([SEQ2IDX[s] for s in seq], device = device)

def idx2seq(idxs):
    return "".join([IDX2SEQ[i] for i in idxs if i != PADDING_VALUE and i != SEQ2IDX[EOS]])

class ProteinDataset(Dataset):
    def __init__(self, file, device = None):
        super().__init__()
        self.device = device
        with open(file) as f:
            seqs = f.readlines()

        list_seqs = map(lambda x: list(x[:-1]) + [EOS], seqs)
        encodedSeqs = map(lambda x: seq2idx(x, self.device), list_seqs)
        self.seqs = list(encodedSeqs)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]

class IterProteinDataset(IterableDataset):
    def __init__(self, file, device = None):
        super().__init__()
        self.file = file
        self.device = device

    def get_str_seqs(self):
        raw_seqs = SeqIO.parse(self.file, "fasta")
        short_seqs = filter(lambda x: len(x) <= 2000, raw_seqs)
        no_BJXZ_seqs = filter(lambda x: all(c not in EXCLUDED_AMINO_ACIDS for c in x), short_seqs)
        str_seqs = map(lambda x: str(x.seq), no_BJXZ_seqs)
        return str_seqs

    def subsample(self, n, outfile):
        str_seqs = self.get_str_seqs()
        for i, _ in enumerate(str_seqs):
            if i % 10000 == 0:
                print(i, end = "\r")

        count = i + 1
        print(f"Saving {n} of {count} proteins...")

        str_seqs = self.get_str_seqs()
        rand_idx = set(random.sample(range(count), k = n))

        with open(outfile, "w") as f:
            for i, seq in enumerate(str_seqs):
                if i % 10000 == 0:
                    print(i, end = "\r")
                if i in rand_idx:
                    f.write(seq + "\n")

    def __iter__(self):
        str_seqs = self.get_str_seqs()
        list_seqs = map(lambda x: list(x) + [EOS], str_seqs)
        encodedSeqs = map(lambda x: seq2idx(x, self.device), list_seqs)
        return encodedSeqs

    def __len__(self):
        return NUM_SEQUENCES

def getProteinDataset(file, device = None):
    extension = file.suffix.lower()
    if extension == ".txt":
        return ProteinDataset(file, device)
    elif extension == ".fasta":
        return IterProteinDataset(file, device)
    else:
        raise ValueError("Unsupported data file.")

def sequenceCollateFn(sequences):
    return torch.nn.utils.rnn.pad_sequence(sequences, padding_value = PADDING_VALUE, batch_first = True)

def getProteinDataLoader(proteinDataset, batch_size = 32):
    return DataLoader(proteinDataset, shuffle = False if isinstance(proteinDataset, IterableDataset) else False, batch_size = batch_size, collate_fn = sequenceCollateFn)
