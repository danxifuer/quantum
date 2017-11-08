from mxnet import gluon
from mxnet.gluon import nn, rnn
import mxnet as mx
from rnn_config import NUM_CLASSES, PREDICT_LEN, SEQ_LEN


class RNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode,
                 num_embed,
                 num_hidden,
                 seq_len, num_layers,
                 dropout=0.0,
                 **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, activation='relu', num_layers=num_layers,
                                   layout='NTC', dropout=dropout, input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers=num_layers, layout='NTC', dropout=dropout,
                                   input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers=num_layers, layout='NTC', dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers=num_layers, layout='NTC', dropout=dropout,
                                   input_size=num_embed)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru" % mode)

            self.fc = nn.Dense(NUM_CLASSES, in_units=num_hidden)
            self.num_hidden = num_hidden
            self.seq_len = seq_len

    def forward(self, inputs, hidden):
        output, hidden = self.rnn(inputs, hidden)
        output = self.drop(output)
        # output = output[:, -PREDICT_LEN:, :]
        output = mx.nd.slice_axis(output, begin=SEQ_LEN - PREDICT_LEN,
                                  end=SEQ_LEN, axis=1)
        decoded = self.fc(output.reshape((-1, self.num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
