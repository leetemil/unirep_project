# Number of sequences in UniRef
NUM_SEQUENCES = 39421231

# The value in the target that should not be counted by cross-entropy loss
INFERENCE_IGNORE_INDEX = -1

# Special tokens in the protein sequences
PAD = "<PAD>"
EOS = "<EOS>"

# Note: <PAD> should be last token since it is not ever guessed
CONTROL_TOKENS = [EOS, PAD]

# Non-ambigious amino acid characters
AMINO_ACIDS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y"]

# All tokens that can be found in sequences, that is the amino acids and control tokens
# Note: The order (Amino acids then control tokens) is important because <PAD> should be the last one
# This ensures that 0 - NUM_INFERENCE_TOKENS are valid inference indices
ALL_TOKENS = AMINO_ACIDS + CONTROL_TOKENS
NUM_TOKENS = len(ALL_TOKENS)

# Number of tokens that the model should be able to guess. -1 because pad should never be guessed
NUM_INFERENCE_TOKENS = NUM_TOKENS - 1

# These characters are ambigious amino acids, whatever that means (I'm not a biologist)
EXCLUDED_AMINO_ACIDS = ["B", "J", "X", "Z"]

# Dictionary mapping a sequence of tokens to a sequence of indices
SEQ2IDX = {a: i for i, a in enumerate(ALL_TOKENS)}

# Index of the pad token
PADDING_VALUE = SEQ2IDX[PAD]

# Dictionary mapping a sequence of indices to a sequence of tokens
IDX2SEQ = {i: a for (a, i) in SEQ2IDX.items()}
