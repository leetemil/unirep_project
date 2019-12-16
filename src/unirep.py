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

    def forward(self, xb, init_hidden):
        # Convert indices to embedded vectors
        embedding = self.embed(xb)

        # Output from RNN is also packed when given packed
        out, last_hidden = self.rnn(embedding, init_hidden)

        # Linear layer to convert from RNN hidden size -> inference tokens scores
        linear = self.lin(out)
        return linear, last_hidden
