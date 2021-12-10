# -*- coding: utf-8 -*-

# This file implements the various model classes we plan on using for this project.
# We define a standard loss function, a "dumb" unsupervised model, a "basic" supervised model,
#   and finally a slightly more advanced supervised model.

# :-)

from torch import nn

from data import DATA_DIMENSION

import torch

# Baseline RNN implementation. We use a 1 layer GRU (with bias) to give a neurel network baseline.
# We use basic mean squared loss and Adam; nothing fancy for this one. The hidden_size is the number
#   of output features we want, in this case, we look only for the close value at the end of the candle.
class Model01(nn.Module):
    # This is a thing we can do, but this dumb class doesn't.
    trainable_hidden_layer = False

    def __init__(self, batch_size):
        super(Model01, self).__init__()

        # This is all we use for the basic model.
        # We disable all of the more advanced features available to get a baseline,
        #   although we keep bias on (no dropout, 1 layer)
        self._gru = nn.GRU(
            input_size = DATA_DIMENSION,
            hidden_size = 1,
            num_layers = 1,
            bias = True,
            batch_first = True, # It's easiest to load our data in this format
            dropout = 0,
            bidirectional = False
        ).double()

    def forward(self, x):
        y, H_out = self._gru(x)

        # We only need to output the last piece of data output by the GRU
        return y[:,-1 ,0]

    def make_optimizer(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    def make_criterion(self):
        self.criterion = nn.MSELoss(reduction = 'mean')

# Slighly improved model. The last model just didn't work.
class Model02(nn.Module):
    # This is a thing we can do, but we still don't.
    trainable_hidden_layer = False

    def __init__(self, batch_size):
        super(Model02, self).__init__()

        # This is a slightly more advanced model, I've enabled dropout and added two layers.
        self._gru = nn.GRU(
            input_size = DATA_DIMENSION,
            hidden_size = 1,
            num_layers = 3,
            bias = True,
            batch_first = True, # It's easiest to load our data in this format
            dropout = 0.25,
            bidirectional = False
        ).double()

    def forward(self, x):
        y, H_out = self._gru(x)

        # We only need to output the last piece of data output by the GRU
        return y[:,-1 ,0]

    def make_optimizer(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    def make_criterion(self):
        self.criterion = nn.MSELoss(reduction = 'mean')

class Model03(nn.Module):
    # This is a thing we can do, but we still don't.
    trainable_hidden_layer = False

    def __init__(self, batch_size):
        super(Model03, self).__init__()

        # This is a slightly more advanced model, I've enabled dropout and added two layers.
        self._gru = nn.GRU(
            input_size = DATA_DIMENSION,
            hidden_size = 1,
            num_layers = 6,
            bias = True,
            batch_first = True, # It's easiest to load our data in this format
            dropout = 0.5,
            bidirectional = False
        ).double()

    def forward(self, x):
        y, H_out = self._gru(x)

        # We only need to output the last piece of data output by the GRU
        return y[:,-1 ,0]

    def make_optimizer(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    def make_criterion(self):
        self.criterion = nn.MSELoss(reduction = 'mean')

class Model04(nn.Module):
    # This is a thing we can do, but we still don't.
    trainable_hidden_layer = False

    def __init__(self, batch_size):
        super(Model04, self).__init__()

        # This is a slightly more advanced model, I've enabled dropout and added two layers.
        self._lstm = nn.LSTM(
            input_size = DATA_DIMENSION,
            hidden_size = 1,
            num_layers = 6,
            bias = True,
            batch_first = True, # It's easiest to load our data in this format
            dropout = 0.5,
            bidirectional = False
        ).double()

    def forward(self, x):
        y, H_out = self._lstm(x)

        # We only need to output the last piece of data output by the GRU
        return y[:,-1 ,0]

    def make_optimizer(self, model):
        self.optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)

    def make_criterion(self):
        self.criterion = nn.MSELoss(reduction = 'mean')

# Unparameterized 'dumb' model.
class AverageModel:
    # Just return the average of the average the open and close values over the series.
    # This is the simplest model possible, we use it as a worst-case baseline.
    def forward(self, X):
        return sum((x[3] + x[4]) / 2 for x in X) / len(x)
