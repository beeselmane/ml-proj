#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data import Database

from cmd import open_dataset_or_exit
from cmd import read_string_option
from cmd import make_granularity
from cmd import find_database
from cmd import date_string

# So we can reconstruct the training state from a save state
from train import do_build_model
from train import TRAIN_DEVICE
from train import SessionState
from train import make_splits
from train import g_state

# So we can rebuild our model
from model import MODEL_CLASSES
from model import AverageModel

from datetime import timedelta
from datetime import datetime
from datetime import timezone

from itertools import chain

import matplotlib.pyplot as plot
import matplotlib

import pickle
import numpy
import torch
import copy
import sys
import os

################################################################################
# Constants/Globals/Initialization Code

# We're going to make some pretty vector graphics I can include in my report.
matplotlib.use('Agg')

################################################################################
# Utility functions

def inflate_model(model_path, meta_file):
    print('Info: Inflating model from disk...', file = sys.stderr)

    if not os.path.exists(model_path):
        print(f'Error: File not found \'{model_path}\'', file = sys.stderr)
        sys.exit(1)

    # First, try and grab our model data.
    model_info = torch.load(model_path)

    g_state.validation_loss = model_info['validation_loss']
    g_state.train_loss = model_info['train_loss']

    # Then, fill in the meta info.
    if meta_file == None:
        if 'meta_path' in model_info:
            meta_file = model_info['meta_path']
        else:
            print('Error: Cannot determine meta file location for model! (Use \'--meta-file\' flag to specify this explicitly)', file = sys.stderr)
            sys.exit(1)

    if not os.path.exists(meta_file):
        print(f'Error: File not found \'{model_path}\'', file = sys.stderr)
        sys.exit(1)

    with open(meta_file, 'rb') as fp:
        meta_object = pickle.load(fp)

    g_state.meta_path = meta_file

    # This is kind of gross, but I blame python for exposing __dict__ to me...
    g_state.__dict__.update(meta_object.__dict__)

    print(f'Info: batch_size = {g_state.batch_size}, sequence_length = {g_state.sequence_length}')
    print(f'Info: granularity = {g_state.granularity}')

    # Recreate the splits used during training
    print('Info: Rebuilding data splits...', file = sys.stderr)

    # Now, we recreate the saved models and optimizer
    model_class = MODEL_CLASSES[meta_object.model_subtype]

    with Database(g_state.database_path) as database:
        variant = Database.Dataset.VARIANT_NORM if model_class.use_normed_data else Database.Dataset.VARIANT_PATCHED

        dataset = open_dataset_or_exit(database, g_state.granularity, variant)

        # Use the same function from the train module.
        make_splits(dataset)

        # We need to grab this column raw for our use later
        g_state.raw_time = dataset.select_all()['time'].to_numpy()

        if hasattr(g_state, 'permutation'):
            g_state.raw_time = g_state.raw_time[g_state.permutation]

    g_state.best_model = do_build_model(model_class)
    g_state.model = do_build_model(model_class)

    g_state.model.optimizer.load_state_dict(model_info['optimizer'])
    g_state.best_model.optimizer = None

    g_state.best_model.load_state_dict(model_info['model_best'])
    g_state.model.load_state_dict(model_info['model_current'])

    # We start in evaluation mode by default.
    g_state.best_model.eval()
    g_state.model.eval()

    # Make this look as close to the train state as possible
    g_state.next_checkpoint = datetime.now()
    g_state.keep_training = False

    print('Info: Inflated model and loaded train session state.', file = sys.stderr)

    return g_state

def show_train_info():
    epochs_per_checkpoint = [len(v) for (k, v) in g_state.train_loss.items()]
    epoch = sum(t for t in epochs_per_checkpoint)

    print('Model info:')

    for cp in g_state.validation_loss.keys():
        print(f'Loss in checkpoint \'{cp}\':')
        print(f'Validation: {g_state.validation_loss[cp][0]} --> {g_state.validation_loss[cp][-1]}')
        print(f'Training:   {g_state.train_loss[cp][0]} --> {g_state.train_loss[cp][-1]}')

    # Don't divide by 0
    if len(epochs_per_checkpoint):
        print(f'Average epochs per checkpoint: {sum(e for e in epochs_per_checkpoint) / len(epochs_per_checkpoint)}')

    print(f'Total checkpoints: {len(g_state.validation_loss.keys())}')
    print(f'Total epochs: {epoch}')

def graph_loss(validation_loss, train_loss, path):
    figure, axes = plot.subplots()

    x_range = range(1, len(validation_loss) + 1)

    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')

    axes.plot(x_range, validation_loss, label = 'Validation Loss')
    axes.plot(x_range, train_loss, label = 'Train Loss')

    axes.legend()

    figure.savefig(path)
    plot.close(figure)

def graph_prediction(times, truth, predictions, averages, path):
    figure, axes = plot.subplots()

    # X label is clear
    axes.set_ylabel('Price (BTC)')

    if predictions is not None:
        axes.scatter(times, predictions, s = 0.5, label = 'Prediction')

    if averages is not None:
        axes.scatter(times, averages, s = 0.5, alpha = 0.25, label = 'Average')

    axes.scatter(times, truth, s = 1, alpha = 0.5, label = 'Actual')

    # Make ticks readable
    axes.set_xticklabels([datetime.utcfromtimestamp(t).strftime('%d-%m-%Y %H:%M:%S') for t in axes.get_xticks()])

    plot.setp(axes.get_xticklabels(), rotation = 45, ha = 'right')

    axes.legend()

    figure.savefig(path, bbox_inches = 'tight')
    plot.close(figure)

def do_graph_predictions(shift, scale):
    def graph_set(s_x, s_y, s_y_p, s_y_m, name):
        order = numpy.argsort(s_x)
        xs = numpy.array(s_x)[order]
        ys = numpy.array(s_y)[order]
        ysp = numpy.array(s_y_p)[order] if s_y_p is not None else None
        sym = numpy.array(s_y_m)[order] if s_y_m is not None else None

        print(f'Info: Making {name} plot...')

        graph_prediction(xs, ys, ysp, sym, f'plot/pred-{name}-{str(g_state.session_id).zfill(4)}-{str(g_state.model_subtype).zfill(2)}-{g_state.batch_size}-{g_state.sequence_length}.pdf')

    def build_info(times, set, offset, length):
        return [
            times[offset:offset + length],
            (set[1][0:length].numpy() * scale) + shift,
            (g_state.best_model(set[0][0:length]).detach().numpy() * scale) + shift,
            (AverageModel().predict(set[0][0:length]).numpy() * scale) + shift
        ]

    train_length = g_state.train_set[0].size()[0]
    validation_length = g_state.validation_set[0].size()[0]
    test_length = g_state.test_set[1].size()[0]

    train_info = build_info(g_state.raw_time, g_state.train_set, 0, train_length)

    graph_set(*train_info, 'train')

    validation_info = build_info(g_state.raw_time, g_state.validation_set, train_length, validation_length)

    graph_set(*validation_info, 'validation')

    test_info = build_info(g_state.raw_time, g_state.test_set, train_length + validation_length, test_length)

    graph_set(*test_info, 'test')

    full_set = [
        numpy.hstack([train_info[0], validation_info[0], test_info[0]]),
        numpy.hstack([train_info[1], validation_info[1], test_info[1]]),
        None,
        numpy.hstack([train_info[3], validation_info[3], test_info[3]])
    ]

    full_x = torch.cat([g_state.train_set[0], g_state.validation_set[0], g_state.test_set[0]], dim = 0)

    print(f'Average Error: {AverageModel().mse(full_x, full_set[1])}')
    print(f'Average Error: {AverageModel().mae(full_x, full_set[1])}')

    graph_set(*full_set, 'all')

def do_make_plots():
    validation_loss = list(chain.from_iterable(v for (k, v) in g_state.validation_loss.items()))
    train_loss = list(chain.from_iterable(v for (k, v) in g_state.train_loss.items()))

    # We need to scale everything back for graphing purposes.
    if g_state.model.use_normed_data:
        with open(f'md/candles_{g_state.granularity}s.pyc', 'rb') as fp:
            metadata = pickle.load(fp)

            scale = metadata['deviation']['eth_btc_close']
            shift = metadata['mean']['eth_btc_close']
    else:
        scale = 1
        shift = 0

    print(f'Best validation loss: {min(validation_loss)}')

    graph_loss(validation_loss, train_loss, f'loss/loss-{str(g_state.session_id).zfill(4)}-{str(g_state.model_subtype).zfill(2)}-{g_state.batch_size}-{g_state.sequence_length}.pdf')

    # Make prediction graphs for train, validation, test
    do_graph_predictions(shift, scale)

def do_test():
    with torch.no_grad():
        test_loss = g_state.model.criterion(g_state.model(g_state.test_set[0]), g_state.test_set[1]).item()

        print(f'Test Loss: {test_loss}')

        return test_loss

################################################################################
# Main Function

def main():
    model_path = read_string_option(sys.argv, '--analyse-model=', None, '', lowercase = False)

    if model_path != '':
        # Don't analyse model, show stats for a given dataset.

        # This option exists to support old save format which doesn't include meta_format
        meta_file = read_string_option(sys.argv, '--meta-file=', None, '', lowercase = False)

        # This fills g_state with the inflated model info
        inflate_model(model_path, meta_file if meta_file != '' else None)

        # Start by showing some basic stats
        if '--show-info' in sys.argv:
            show_train_info()

        do_test()

        do_make_plots()
    elif ('--avg-model'):
        model = AverageModel
    else:
        # Reconstruct training state and things.

        # Select a given candle granularity
        granularity = make_granularity(sys.argv)

        # Find database
        database_path = find_database(sys.argv)

        # Parse database file to use and open database
        with Database(database_path) as database:
            return

if __name__ == '__main__':
    main()
