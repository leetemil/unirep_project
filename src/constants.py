from collections import OrderedDict

# Number of sequences in UniRef
NUM_SEQUENCES = 39421231

# The value in the target that should not be counted by cross-entropy loss
INFERENCE_IGNORE_INDEX = -1

UNIREP_VOCAB = OrderedDict([
    ("<pad>", 0),
    ("M", 1),
    ("R", 2),
    ("H", 3),
    ("K", 4),
    ("D", 5),
    ("E", 6),
    ("S", 7),
    ("T", 8),
    ("N", 9),
    ("Q", 10),
    ("C", 11),
    ("U", 12),
    ("G", 13),
    ("P", 14),
    ("A", 15),
    ("V", 16),
    ("I", 17),
    ("F", 18),
    ("Y", 19),
    ("W", 20),
    ("L", 21),
    ("O", 22),
    ("X", 23),
    ("Z", 23),
    ("B", 23),
    ("J", 23),
    ("<cls>", 24),
    ("<sep>", 25)])

CLS = "<cls>"
SEP = "<sep>"

NUM_TOKENS = 26

# These characters are ambigious amino acids, whatever that means (I'm not a biologist)
EXCLUDED_AMINO_ACIDS = ["B", "J", "X", "Z"]

# Dictionary mapping a sequence of tokens to a sequence of indices
SEQ2IDX = UNIREP_VOCAB

# Index of the pad token
PADDING_VALUE = SEQ2IDX["<pad>"]

# Dictionary mapping a sequence of indices to a sequence of tokens
IDX2SEQ = {i: a for (a, i) in SEQ2IDX.items()}
