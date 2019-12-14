import warnings
from collections import namedtuple
from typing import List, Tuple
import numbers

import torch
from torch import nn
from torch import Tensor
from torch.nn import Parameter
from torch import jit

LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

# From https://github.com/guillitte/pytorch-sentiment-neuron/blob/master/models.py
class mLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # TODO: Consider bias or not?
        self.wx = nn.Linear(input_size, 4 * hidden_size, bias = True)
        self.wh = nn.Linear(hidden_size, 4 * hidden_size, bias = True)
        self.wmx = nn.Linear(input_size, hidden_size, bias = True)
        self.wmh = nn.Linear(hidden_size, hidden_size, bias = True)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = state
        m = self.wmx(input) * self.wmh(hx)
        gates = self.wx(input) + self.wh(m)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)

class LSTMLayer(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs = input.unbind(1)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

def script_mlstm(input_size, hidden_size, num_layers = 1, bias=True,
                batch_first=True, dropout=False, bidirectional=False):
    '''Returns a ScriptModule that implements an mLSTM.'''

    # The following are not implemented.
    assert bias
    assert batch_first

    if bidirectional:
        stack_type = StackedLSTM2
        layer_type = BidirLSTMLayer
        dirs = 2
    elif dropout:
        stack_type = StackedLSTMWithDropout
        layer_type = LSTMLayer
        dirs = 1
    else:
        stack_type = StackedLSTM
        layer_type = LSTMLayer
        dirs = 1

    return stack_type(num_layers, layer_type,
                      first_layer_args=[mLSTMCell, input_size, hidden_size],
                      other_layer_args=[mLSTMCell, hidden_size * dirs,
                                        hidden_size])

def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)

class StackedLSTM(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


# Differs from StackedLSTM in that its forward method takes
# List[List[Tuple[Tensor,Tensor]]]. It would be nice to subclass StackedLSTM
# except we don't support overriding script methods.
# https://github.com/pytorch/pytorch/issues/10733
class StackedLSTM2(jit.ScriptModule):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM2, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[List[Tuple[Tensor, Tensor]]]) -> Tuple[Tensor, List[List[Tuple[Tensor, Tensor]]]]
        # List[List[LSTMState]]: The outer list is for layers,
        #                        inner list is for directions.
        output_states = jit.annotate(List[List[Tuple[Tensor, Tensor]]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


class StackedLSTMWithDropout(jit.ScriptModule):
    # Necessary for iterating through self.layers and dropout support
    __constants__ = ['layers', 'num_layers']

    def __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTMWithDropout, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)
        # Introduces a Dropout layer on the outputs of each LSTM layer except
        # the last layer, with dropout probability = 0.4.
        self.num_layers = num_layers

        if (num_layers == 1):
            warnings.warn("dropout lstm adds dropout layers after all but last "
                          "recurrent layer, it expects num_layers greater than "
                          "1, but got num_layers = 1")

        self.dropout_layer = nn.Dropout(0.4)

    @jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            # Apply the dropout layer except the last layer
            if i < self.num_layers - 1:
                output = self.dropout_layer(output)
            output_states += [out_state]
            i += 1
        return output, output_states

def test_script_rnn_layer(seq_len, batch, input_size, hidden_size):
    inp = torch.randn(seq_len, batch, input_size)
    state = LSTMState(torch.randn(batch, hidden_size),
                      torch.randn(batch, hidden_size))
    rnn = LSTMLayer(LSTMCell, input_size, hidden_size)
    out, out_state = rnn(inp, state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))
    for lstm_param, custom_param in zip(lstm.all_weights[0], rnn.parameters()):
        assert lstm_param.shape == custom_param.shape
        with torch.no_grad():
            lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (out_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_stacked_rnn(seq_len, batch, input_size, hidden_size,
                            num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = [LSTMState(torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size))
              for _ in range(num_layers)]
    rnn = script_lstm(input_size, hidden_size, num_layers)
    out, out_state = rnn(inp, states)
    custom_state = flatten_states(out_state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers)
    lstm_state = flatten_states(states)
    for layer in range(num_layers):
        custom_params = list(rnn.parameters())[4 * layer: 4 * (layer + 1)]
        for lstm_param, custom_param in zip(lstm.all_weights[layer],
                                            custom_params):
            assert lstm_param.shape == custom_param.shape
            with torch.no_grad():
                lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_stacked_bidir_rnn(seq_len, batch, input_size, hidden_size,
                                  num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = [[LSTMState(torch.randn(batch, hidden_size),
                         torch.randn(batch, hidden_size))
               for _ in range(2)]
              for _ in range(num_layers)]
    rnn = script_lstm(input_size, hidden_size, num_layers, bidirectional=True)
    out, out_state = rnn(inp, states)
    custom_state = double_flatten_states(out_state)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
    lstm_state = double_flatten_states(states)
    for layer in range(num_layers):
        for direct in range(2):
            index = 2 * layer + direct
            custom_params = list(rnn.parameters())[4 * index: 4 * index + 4]
            for lstm_param, custom_param in zip(lstm.all_weights[index],
                                                custom_params):
                assert lstm_param.shape == custom_param.shape
                with torch.no_grad():
                    lstm_param.copy_(custom_param)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    assert (out - lstm_out).abs().max() < 1e-5
    assert (custom_state[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_state[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_stacked_lstm_dropout(seq_len, batch, input_size, hidden_size,
                                     num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = [LSTMState(torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size))
              for _ in range(num_layers)]
    rnn = script_lstm(input_size, hidden_size, num_layers, dropout=True)

    # just a smoke test
    out, out_state = rnn(inp, states)


def test_script_stacked_lnlstm(seq_len, batch, input_size, hidden_size,
                               num_layers):
    inp = torch.randn(seq_len, batch, input_size)
    states = [LSTMState(torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size))
              for _ in range(num_layers)]
    rnn = script_lnlstm(input_size, hidden_size, num_layers)

    # just a smoke test
    out, out_state = rnn(inp, states)

# test_script_rnn_layer(5, 2, 3, 7)
# test_script_stacked_rnn(5, 2, 3, 7, 4)
# test_script_stacked_bidir_rnn(5, 2, 3, 7, 4)
# test_script_stacked_lstm_dropout(5, 2, 3, 7, 4)
# test_script_stacked_lnlstm(5, 2, 3, 7, 4)
