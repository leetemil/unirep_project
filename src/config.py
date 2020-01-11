import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description = "Runs training of the UniRep model.")

parser.add_argument("data", type = Path, help = "File to load training data from")
parser.add_argument("validation", type = Path, help = "File to load validation data from")
parser.add_argument("--epochs", type = int, default = 1000, help = "Number of epochs to train for.")
parser.add_argument("--batch_size", type = int, default = 1024, help = "Size of each batch.")
parser.add_argument("--truncation_window", type = int, default = 256, help = "Truncation window of truncated backpropogation through time")
parser.add_argument("--rnn", type = str, choices = ["LSTM", "GRU", "mLSTM"], default = "mLSTM", help = "Which type of recurrent neural network to use.")
parser.add_argument("--embed_size", type = int, default = 10, help = "Dimension of the embedded vectors.")
parser.add_argument("--hidden_size", type = int, default = 1024, help = "Size of the RNNs hidden state.")
parser.add_argument("--num_layers", type = int, default = 1, help = "Number of stacked RNN layers.")
parser.add_argument("--print_every", type = int, default = 10, help = "How many batches between printing training progress.")
parser.add_argument("--save_every", type = int, default = 10, help = "How many batches between saving the model.")
parser.add_argument("--load_path", type = Path, default = Path(), help = "File to load an existing model from.")
parser.add_argument("--save_path", type = Path, default = Path(), help = "File to save the model to. It will be overwritten if it exists already.")

config = parser.parse_args()
