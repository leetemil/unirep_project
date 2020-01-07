import torch
from torch import nn

from constants import *

from mlstm import mLSTM

class UniRep(nn.Module):
    def __init__(self, rnn_type, embed_size, hidden_size, num_layers):
        super().__init__()

        # Define parameters
        self.rnn_type = rnn_type
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define layers
        self.embed = nn.Embedding(NUM_TOKENS, self.embed_size, padding_idx = PADDING_VALUE)

        self.lin = nn.Linear(self.hidden_size, NUM_INFERENCE_TOKENS)

        if rnn_type == "mLSTM":
            self.rnn = mLSTM(self.embed_size, self.hidden_size, num_layers = self.num_layers)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(self.embed_size, self.hidden_size, num_layers = self.num_layers, batch_first = True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(self.embed_size, self.hidden_size, num_layers = self.num_layers, batch_first = True)
        else:
            ValueError("Unsupported RNN type.")

    def forward(self, xb, hidden, mask):
        # Convert indices to embedded vectors
        embedding = self.embed(xb)

        if hidden is not None:
            if self.rnn_type == "mLSTM":
                hidden = [(h.detach(), c.detach()) for h, c in hidden]
            elif self.rnn_type == "LSTM":
                hidden = [h.detach() for h in hidden]
            elif self.rnn_type == "GRU":
                hidden = hidden.detach()

        if self.rnn_type == "mLSTM":
            out, last_hidden = self.rnn(embedding, hidden, mask)
        else:
            out, last_hidden = self.rnn(embedding, hidden)

        # Linear layer to convert from RNN hidden size -> inference tokens scores
        linear = self.lin(out)
        log_likelihoods = nn.functional.log_softmax(linear, dim = 2)
        return log_likelihoods, last_hidden

    def get_representations(self, xb, mask):
        with torch.no_grad():
            embedding = self.embed(xb)
            hidden = None
            if self.rnn_type == "mLSTM":
                out, _ = self.rnn(embedding, hidden, mask)
            else:
                out, _ = self.rnn(embedding, hidden)

            mask = mask.unsqueeze(-1)
            masked_out = out * mask
            representations = masked_out.sum(dim = 1) / mask.sum(dim = 1)
            return representations

    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())
        num_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (f"UniRep summary:\n"
                f"  RNN type:    {type(self.rnn).__name__}\n"
                f"  Embed size:  {self.embed_size}\n"
                f"  Hidden size: {self.hidden_size}\n"
                f"  Layers:      {self.num_layers}\n"
                f"  Parameters:  {num_params:,}\n")
