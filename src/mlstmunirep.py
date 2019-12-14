import torch
from torch import nn

from constants import *

from mlstm import script_mlstm

class UniRep_mLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super().__init__()

        # Define parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.embed = nn.Embedding(NUM_TOKENS, self.embed_size, padding_idx = PADDING_VALUE)
        self.rnn = script_mlstm(self.embed_size, self.hidden_size, num_layers = self.num_layers, batch_first = True)
        self.lin = nn.Linear(self.hidden_size, NUM_INFERENCE_TOKENS)

    def forward(self, xb, xb_lens, init_hidden):
        # This is important to do for DataParallel. We need this length when padding the packed sequence again
        longest_length = xb.size(1)

        # Convert indices to embedded vectors
        embedding = self.embed(xb)

        # Output from RNN is also packed when given packed
        out, _ = self.rnn(embedding, init_hidden)

        # Linear layer to convert from RNN hidden size -> inference tokens scores
        linear = self.lin(out)
        return linear
