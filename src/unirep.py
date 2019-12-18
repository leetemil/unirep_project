import torch
from torch import nn

from constants import *

from mlstm import mLSTM

class UniRep(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, use_mlstm=False):
        super().__init__()

        # mlstm flag
        self.use_mlstm = use_mlstm

        # Define parameters
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.embed = nn.Embedding(NUM_TOKENS, self.embed_size, padding_idx = PADDING_VALUE)

        self.lin = nn.Linear(self.hidden_size, NUM_INFERENCE_TOKENS)

        if use_mlstm:
            self.rnn = mLSTM(self.embed_size, self.hidden_size, num_layers = self.num_layers)

        else:
            self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers = self.num_layers, batch_first = True)


    def forward(self, xb, hidden):
        # Convert indices to embedded vectors
        embedding = self.embed(xb)

        if self.use_mlstm:
            if hidden is not None:
                hidden = [(h.detach(), c.detach()) for h, c in hidden]
            out, last_hidden = self.rnn(embedding, hidden)
        else:
            out, last_hidden = self.rnn(embedding, [h.detach() for h in hidden] if hidden else None)

        # Linear layer to convert from RNN hidden size -> inference tokens scores
        linear = self.lin(out)
        return linear, last_hidden

    def get_representation(self, xb, device):
        with torch.no_grad():
            embedding = self.embed(xb)
            out, _ = self.rnn(embedding, self.init_hidden(len(xb), device))

            return torch.mean(out, dim=1)
