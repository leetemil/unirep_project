import torch
from torch import nn

from constants import *

class UniRep(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super().__init__()

        # Define parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.embed = nn.Embedding(NUM_TOKENS, self.embed_size, padding_idx = PADDING_VALUE)
        self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers = self.num_layers, batch_first = True)
        self.lin = nn.Linear(self.hidden_size, NUM_INFERENCE_TOKENS)

    def forward(self, xb, xb_lens):
        # This is important to do for DataParallel. We need this length when padding the packed sequence again
        longest_length = xb.size(1)

        # Convert indices to embedded vectors
        embedding = self.embed(xb)

        # Convert padded batch to a packed sequence. This makes it so the LSTM does not compute anything for the pad values
        packed_seqs = nn.utils.rnn.pack_padded_sequence(embedding, xb_lens, batch_first = True, enforce_sorted = False)

        # Output from RNN is also packed when given packed
        packed_out, _ = self.rnn(packed_seqs)

        # Convert packed sequence back into padded batch. Using the longest length from the original batch is important
        # as when using DataParallel, the padded batch may be longer than the longest sequence in the batch,
        # since the longest sequence may have been given to another GPU. Doing it this way ensures that all GPUs produce
        # an output batch that is the same dimensions.
        out, lens = nn.utils.rnn.pad_packed_sequence(packed_out, total_length = longest_length, batch_first = True)

        # Linear layer to convert from RNN hidden size -> inference tokens scores
        linear = self.lin(out)
        return linear
