INFERENCE_IGNORE_INDEX = -1

PAD = "<PAD>"
EOS = "<EOS>"
AMINO_ACIDS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y"]
CONTROL_TOKENS = [EOS, PAD]
ALL_TOKENS = AMINO_ACIDS + CONTROL_TOKENS

EXCLUDED_AMINO_ACIDS = "BJXZ"
NUM_TOKENS = len(ALL_TOKENS)
NUM_INFERENCE_TOKENS = NUM_TOKENS - 1

SEQ2IDX = {a: i for i, a in enumerate(ALL_TOKENS)}
IDX2SEQ = {i: a for (a, i) in SEQ2IDX.items()}

PADDING_VALUE = SEQ2IDX[PAD]
