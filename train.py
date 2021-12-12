#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file trains a model on a requested dataset.
# We want to include both the basic models and the more complex models.
#
# This way, we can centralize everything related to training to this one file,
#   and spawning a trainer for more complex tasks can be done with command line
#   flags instead of separate commands entirely.
#
# Thus, everything related to training exists here and in model.py.
# Model parameters, training results, etc. can all be put into the sqlite database.

from data import DATA_DIMENSION
from data import Database

# Note that these should really just be in a list in the model module...
from model import MODEL_CLASSES
from model import AverageModel

from cmd import open_dataset_or_exit
from cmd import read_string_option
from cmd import make_granularity
from cmd import parse_date_arg
from cmd import find_database

from datetime import timedelta
from datetime import datetime
from datetime import timezone

# I could write this constant again, but this doesn't hurt...
from hist import SEC_PER_DAY

import signal
import pickle
import torch
import copy
import sys
import os

# Here, we need to do a few things:
# 1. Split into train/validation/test sets.
#    - Here, I'm basically going to pick 2019/2020 as train data, 2021-1/2021-10 as validation data,
#        and anything past 31-10-2021 as test data.
# 2. Select a model.
# 3. Loop training.
#    - If we do a supervised model, we need to run on train batches, do gradient descent,
#      run validation every so often, save model with best validation error, and keep going until convergence.
# 4. Save parameters somewhere
#    - Pass in a model directory, train id #, and save things as '${model_dir}/modelXX_${id}_${datetime.isotime()}.pth'
#    - We can save every few minutes I suppose, these things shouldn't take up a ton of room I assume... maybe 30 minutes?
#    - When saving, save `BEST_VALIDATION` and `current` parameters. Only save `best_validation` if it's become dirty since
#      the last save.
#    - So, we save as 'models/modelXX_nn_BEST_VALIDATION_isotime.pth' or 'models/model_nn_CURRENT_isotime.pth'
#      where XX is the model number. Whenever we make changes to the model structure, we need to create a new class if
#      we have saved parameters for a previous model.

# pytorch recommends using `best_model_state = deepcopy(model.state_dict())` to copy the state_dict of a model.
# We can save using `torch.save(model, ${PATH})` and load using `model = torch.load(${PATH}); model.eval()`

# Or, we can save checkpoints as such:
#
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             ...
#             }, PATH)
#
# And load as:
#
# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)
#
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
#
# model.eval()
# # - or -
# model.train()


# For loading on GPU:
#device = torch.device("cuda")
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model

# Arguments to process:
#
# --split-style={random,fixed}
# --sequence-length=nn
#
# if (split-style == fixed)
#   --train-start=datetime --train-end=datetime
#   --validation-start=dataetime --validation-end=datetime
#   --test-start=datetime --test-end=datetime
#
# --type={unparameterized,parameterized} --model=XX
#
# --model-dir=model
#
# if (type == parameterized)
#   --batch-size=bb
#
# --session-id=ss
#
# --resume-from='model/train_info.nn.dat'

# ./train.py --granularity=86400 --db-loc=coinbase.sqlite
#     -split-style=fixed --sequence-length=7
#     --model-type=parameterized --model=01
#     --train-start=1-1-2019 --train-end=1-1-2021
#     --validation-start=1-1-2021 --validation-end=1-10-2021
#     --test-start=1-10-2021 --test-end=20-11-2021
#     --model-dir=model
#     --batch-size=12
#     --session-id=01

################################################################################
# Data Structures

class SessionState:
    # This class is purposefully empty.
    # I'm not sure of the `pythonic' way of doing this, but basically attributes
    #   are added to the instance we keep below at runtime. I suppose we don't
    #   technically need a class for this, but it makes accesses pretty I think..
    def __init__(self):
        pass

# We keep a global state around so we can access it from the signal handler below.
# This is mainly so we can save on request
g_state = SessionState()

################################################################################
# Setup

# We save during training as each checkpoint elapses
CHECKPOINT_DELTA = timedelta(minutes = 15)

# This is the device we train on
TRAIN_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Info: Using device \'{TRAIN_DEVICE}\'', file = sys.stderr)

# This signal handler will be called at SIGINT
def sigint_handler(sig, frame):
    signal.signal(signal.SIGINT, default_sigint_handler)

    action = input('Interrupted. What action should I take? ').lower()

    if action in ['q', 'quit']:
        should_save = input('Would you like to save first (Y/n)? ').lower()

        if should_save in ['y', 'yes']:
            do_save_all(user_info = 'FINAL')

            print('Okay. Exiting now.')
            sys.exit(0)
        elif should_save in ['n', 'no']:
            print('Okay. Exiting now.')
            sys.exit(0)
        else:
            print('Unrecognized input. Will not quit.')
    elif action in ['s', 'save']:
        do_save_all(user_info = 'MANUAL')
    elif action in ['t', 'test']:
        print('Okay. Running test.')

        g_state.keep_training = False
    elif action in ['i', 'info']:
        checkpoint = g_state.next_checkpoint.isoformat()

        epochs_per_checkpoint = [len(v) for (k, v) in g_state.train_loss.items() if k != checkpoint]
        epoch = sum(t for t in epochs_per_checkpoint) + len(g_state.train_loss[checkpoint])

        print('Train info:')
        print(f'Next checkpoint: {checkpoint}')
        print('')

        # Guard against bad indecies
        if len(g_state.validation_loss[g_state.next_checkpoint.isoformat()]):
            for cp in g_state.validation_loss.keys():
                print(f'First loss in checkpoint \'{cp}\':')
                print(f'Validation: {g_state.validation_loss[cp][0]}')
                print(f'Training:   {g_state.train_loss[cp][0]}')

            print('')
            print(f'Last validation loss: {g_state.validation_loss[g_state.next_checkpoint.isoformat()][-1]}')
            print(f'Last train loss: {g_state.train_loss[g_state.next_checkpoint.isoformat()][-1]}')
            print(f'Best validation loss: {g_state.validation_best}')
            print('')

        # Don't divide by 0
        if len(epochs_per_checkpoint):
            print(f'Average epochs per checkpoint: {sum(e for e in epochs_per_checkpoint) / len(epochs_per_checkpoint)}')

        print(f'Total epochs until now: {epoch}')
    elif action == 'xxx':
        # This option helps me debug.
        sys.exit(0)
    else:
        print('Unrecognized action. Continuing...')

    signal.signal(signal.SIGINT, sigint_handler)

################################################################################
# Argument processing

def read_numeric_option(args, prefix):
    flags = [s for s in args if s.lower().startswith(prefix)]

    if not flags:
        print(f'Error: No \'{prefix[0:-1]}\' option specified!', file = sys.stderr)

        sys.exit(1)

    if len(flags) != 1:
        print(f'Error: Ambiguous \'{prefix[0:-1]}\' option!', file = sys.stderr)
        sys.exit(1)

    try:
        return int(flags[0][len(prefix):])
    except ValueError as error:
        print(f'Error: Unacceptable \'{prefix[0:-1]}\' value!', file = sys.stderr)
        sys.exit(1)

################################################################################
# Utility Functions

def check_dir(dir_path):
    if os.path.isdir(dir_path):
        return True
    elif os.path.exists(dir_path):
        return False

    os.mkdir(dir_path, mode = 0o755)

    return True

def do_build_model(Model):
    instance = Model(g_state.batch_size).to(TRAIN_DEVICE)

    # This needs to happen after the call to to() above.
    instance.make_optimizer(instance)
    instance.make_criterion()

    print('Info: Instantiated model.', file = sys.stderr)

    return instance

################################################################################
# Core Logic

# We need to be able to save everything going on
def do_save_all(user_info):
    utc_isotime = datetime.utcnow().replace(tzinfo = timezone.utc).isoformat()

    save_path = f'{g_state.data_dir}/model{str(g_state.model_subtype).zfill(2)}_{str(g_state.session_id).zfill(4)}_{user_info}_{utc_isotime}.pth'

    if os.path.exists(save_path):
        print(f'Warning: Default path \'{save_path}\' exists!')
        print(f'Info: Will save to tmp file instead.')

        save_path = save_path + '.tmp'

    torch.save({
        'validation_loss'   : g_state.validation_loss,
        'train_loss'        : g_state.train_loss,
        'model_current'     : g_state.model.state_dict(),
        'model_best'        : g_state.model_best.state_dict(),
        'optimizer'         : g_state.model.optimizer.state_dict(),
        'meta_path'         : g_state.meta_path
    }, save_path)

    print('Info: Saved training progress.', file = sys.stderr)

# Just dump metadata here
def record_meta():
    utc_isotime = datetime.utcnow().replace(tzinfo = timezone.utc).isoformat()

    g_state.meta_path = f'{g_state.data_dir}/session_{str(g_state.session_id).zfill(4)}_{utc_isotime}.pyc'

    if os.path.exists(g_state.meta_path):
        print(f'Error: Refusing to overwrite file at path \'{g_state.meta_path}\'!', file = sys.stderr)
        sys.exit(1)

    # We want to save the global state, but certain runtime generated fields are uneeded.
    # We create a deep copy we can modify before writing it.
    meta_object = copy.deepcopy(g_state)

    # We don't want to save global state variables that we can recreate dynamically
    meta_object.__dict__.pop('next_checkpoint', None)
    meta_object.__dict__.pop('keep_training', None)
    meta_object.__dict__.pop('model', None)

    # Don't save our data partitions (or maybe we actually should save these?)
    # Saving these would allow random splits, but it's not trivial to implement.
    meta_object.__dict__.pop('train_set', None)
    meta_object.__dict__.pop('validation_set', None)
    meta_object.__dict__.pop('test_set', None)

    # Don't save these here, we save them with the model itself.
    meta_object.__dict__.pop('validation_loss', None)
    meta_object.__dict__.pop('train_loss', None)

    # Same as above.
    meta_object.__dict__.pop('validation_best', None)
    meta_object.__dict__.pop('model_best', None)

    with open(g_state.meta_path, 'xb') as fp:
        pickle.dump(meta_object, fp)

    print(f'Info: Saved session metadata to file: \'{g_state.meta_path}\'', file = sys.stderr)

def do_train():
    def do_batch(model, batch_x, batch_y):
        model.optimizer.zero_grad()

        loss = model.criterion(model(batch_x), batch_y)
        loss.backward()

        model.optimizer.step()

        return loss.item()

    def do_validation(model, validation_x, validation_y):
        with torch.no_grad():
            model.eval()

            # This is all we need, get it and get out.
            loss = model.criterion(model(validation_x), validation_y)

            model.train()

        return loss.item()

    print('Info: Starting training now.', file = sys.stderr)
    print(f'Note: Send SIGINT to this process (pid = {os.getpid()}) for interactive options.', file = sys.stderr)

    train_samples = g_state.train_set[0].size()[0]
    batch_count = train_samples / g_state.batch_size

    if not hasattr(g_state, 'validation_loss'):
        # We need to initialize these here
        g_state.validation_loss = {}
        g_state.train_loss = {}

        g_state.model_best = copy.deepcopy(g_state.model)
        g_state.validation_best = sys.maxsize # This is a very large number..

    while g_state.keep_training:
        g_state.next_checkpoint = datetime.now() + CHECKPOINT_DELTA

        print(f'Info: Next checkpoint at: {g_state.next_checkpoint.isoformat()}', file = sys.stderr)

        # We track these so we can make graphs later, and so we can continue training from disk.
        g_state.validation_loss[g_state.next_checkpoint.isoformat()] = []
        g_state.train_loss[g_state.next_checkpoint.isoformat()] = []

        while g_state.keep_training and datetime.now() < g_state.next_checkpoint:
            permutation = torch.randperm(train_samples, device = TRAIN_DEVICE)
            round_loss = 0

            for b in range(0, train_samples, g_state.batch_size):
                batch_split = permutation[b : b + g_state.batch_size]

                round_loss += do_batch(g_state.model, g_state.train_set[0][batch_split], g_state.train_set[1][batch_split])

            validation_loss = do_validation(g_state.model, *g_state.validation_set)

            g_state.train_loss[g_state.next_checkpoint.isoformat()].append(round_loss / batch_count)
            g_state.validation_loss[g_state.next_checkpoint.isoformat()].append(validation_loss)

            if validation_loss < g_state.validation_best:
                g_state.model_best = copy.deepcopy(g_state.model)
                g_state.validation_best = validation_loss

        print(f'Info: Checkpoint reached. Best validation loss: {g_state.validation_best}', file = sys.stderr)

        # Save when we reach a checkpoint
        do_save_all(user_info = 'CHECKPOINT')

    print('Info: Stopped training.', file = sys.stderr)
    do_save_all(user_info = 'TRAIN_DONE')

def do_test():
    def test(model, x, y):
        with torch.no_grad():
            model.eval()

            # This is all we need, get it and get out.
            loss = model.criterion(model(x), y)

            model.train()

        return loss.item()

    print('Info: Running model test!', file = sys.stderr)

    print(f'Loss over test set (current model): {test(g_state.model, *g_state.test_set)}')
    print(f'Loss over test set (best model): {test(g_state.model_best, *g_state.test_set)}')

    while True:
        continue_training = input('Keep training (Y/n)? ').lower()

        if continue_training in ['y', 'yes']:
            print('Okay, resuming...')
            return True
        elif continue_training in ['n', 'no']:
            print('Okay, will exit.')
            return False
        else:
            print('Unrecognized option. ', end = '')

def make_splits(dataset):
    # Process data in a specified range
    def process(range_in):
        raw_set = dataset.select_range(*range_in, inclusive = False)

        if raw_set.empty:
            return None

        patched_raw = fixup(raw_set)

        return unwrap(patched_raw, g_state.sequence_length)

    # Unwrap into a mutli-dimensional `length` buffer
    def unwrap(raw, length):
        # We reserve the last value in the sequence as the y value
        x_len = length - 1
        y_len = length - x_len # = 1

        x_tensor = torch.zeros([raw.shape[0] - length, x_len, DATA_DIMENSION], dtype = torch.float64)
        y_tensor = torch.zeros([raw.shape[0] - length, y_len, 1], dtype = torch.float64)

        # Oh god the data copies are killing me...........
        # I'm not sure how to optimize elsewise though, I just need to train these dang networks...
        # This works at least, or I'm 99% sure it does...
        for i in range(raw.shape[0] - length):
            # Grab the first values in the sequence for x, the last values for y
            x_tensor[i] = torch.from_numpy(raw[i:i + x_len].to_numpy())

            # For y, we only want the value we want to predict, that is, the close value at the end of the interval
            y_tensor[i] = torch.from_numpy(raw['eth_btc_close'][i + x_len:(i + x_len) + y_len].to_numpy())

        # y is only one dimensional, this eases some things down the line...
        return [x_tensor.to(TRAIN_DEVICE), y_tensor.view(raw.shape[0] - length).to(TRAIN_DEVICE)]

    # For now, we simply convert time from unix timestamp to time of day.
    def fixup(raw):
        raw['time'] %= SEC_PER_DAY
        return raw

    # First, we need to grab the raw data in each split.
    if g_state.split_style == 'random':
        # Random. Just randomly split everything.
        # Actually, I don't have a way of tracking this...
        # We can only support fixed splits for now...

        if not hasattr(g_state, 'permutation'):
            g_state.train_split = 0.7
            g_state.validation_split = 0.2
            g_state.test_split = 0.1

        raw_set = dataset.select_all()

        if raw_set.empty:
            print('Error: No data to use!', file = sys.stderr)
            sys.exit(1)

        # Grab everything
        whole_set = unwrap(fixup(raw_set), g_state.sequence_length)
        sample_count = whole_set[0].size()[0]

        train_length = int((sample_count * g_state.train_split) / g_state.batch_size) * g_state.batch_size
        validation_length = int((sample_count * g_state.validation_split) / g_state.batch_size) * g_state.batch_size

        if not hasattr(g_state, 'permutation'):
            g_state.permutation = torch.randperm(sample_count)

        g_state.train_set = [t[g_state.permutation[0:train_length]] for t in whole_set]
        g_state.validation_set = [t[g_state.permutation[train_length:train_length + validation_length]] for t in whole_set]
        g_state.test_set = [t[g_state.permutation[train_length + validation_length:]] for t in whole_set]

        print(f'Info: Train set length: {g_state.train_set[0].size()[0]}', file = sys.stderr)
        print(f'Info: Validation set length: {g_state.validation_set[0].size()[0]}', file = sys.stderr)
        print(f'Info: Test set length: {g_state.test_set[0].size()[0]}', file = sys.stderr)
    else:
        g_state.train_set = process(g_state.train_range)

        if g_state.train_set == None:
            print('Error: Failed to process train set!', file = sys.stderr)
            sys.exit(1)

        if g_state.train_set[0].size()[0] % g_state.batch_size:
            print(f'Warning: Batch size \'{g_state.batch_size}\' is not a multiple of train set length \'{g_state.train_set[0].size()[0]}\'', file = sys.stderr)

            trunc_slice = slice(0, -int(g_state.train_set[0].size()[0] % g_state.batch_size))
            g_state.train_set = [s[trunc_slice] for s in g_state.train_set]

            print(f'Info: Truncated to \'{g_state.train_set[0].size()[0]}\' entries.', file = sys.stderr)

        g_state.validation_set = process(g_state.validation_range)

        if g_state.validation_set == None:
            print('Error: Failed to process validation set!', file = sys.stderr)
            sys.exit(1)

        g_state.test_set = process(g_state.test_range)

        if g_state.test_set == None:
            print('Error: Failed to process test set!', file = sys.stderr)
            sys.exit(1)

        print('Info: Built train/validation/test splits.', file = sys.stderr)

################################################################################
# Main Function

def main():
    # Get our session ID for saving
    g_state.session_id = read_numeric_option(sys.argv, '--session-id=')

    # Select a given candle granularity
    g_state.granularity = make_granularity(sys.argv)

    # Find database
    g_state.database_path = find_database(sys.argv)

    # Create the train/validate/test splits
    g_state.split_style = read_string_option(sys.argv, '--split-style=', ['random', 'fixed'], 'random')

    # How much data should we feed at once?
    g_state.sequence_length = read_numeric_option(sys.argv, '--sequence-length=')

    if g_state.split_style == 'fixed':
        # Fixed. Look for command options, `make_splits` will look for these options later.
        g_state.train_range = [parse_date_arg(sys.argv, '--train-start='), parse_date_arg(sys.argv, '--train-end=')]
        g_state.validation_range = [parse_date_arg(sys.argv, '--validation-start='), parse_date_arg(sys.argv, '--validation-end=')]
        g_state.test_range = [parse_date_arg(sys.argv, '--test-start='), parse_date_arg(sys.argv, '--test-end=')]

        # This is the only error checking we do until we've opened the dataset later
        if sum(t == None for t in (g_state.train_range + g_state.validation_range + g_state.test_range)) > 0:
            print('Error: Unable to parse option for split ranges!', file = sys.stderr)

            sys.exit(1)

    # Determine and create the model we want to train
    g_state.model_type = read_string_option(sys.argv, '--model-type=', ['unparameterized', 'parameterized'], 'parameterized')

    if g_state.model_type == 'parameterized':
        g_state.batch_size = read_numeric_option(sys.argv, '--batch-size=')
        g_state.model_subtype = read_numeric_option(sys.argv, '--model=')

        try:
            model_class = MODEL_CLASSES[g_state.model_subtype]
        except IndexError as error:
            print(f'Error: Unrecognized model subtype \'{model_subtype}\'!', file = sys.stderr)
            sys.exit(1)

        # Instantiate the model for this session here.
        # Note this function will access g_state.batch_size
        g_state.model = do_build_model(model_class)
    else:
        # This is a special case we need to handle later...
        # This is a non-parametreic model we use as a baseline.
        pass

    # Make sure our data directory exists.
    g_state.data_dir = os.path.normpath(read_string_option(sys.argv, '--model-dir=', None, 'model', lowercase = False))

    # This function ensures our data directory exists, and creates it if not.
    check_dir(g_state.data_dir)

    # Start doing real work
    with Database(g_state.database_path) as database:
        variant = Database.Dataset.VARIANT_NORM if g_state.model.use_normed_data else Database.Dataset.VARIANT_PATCHED

        dataset = open_dataset_or_exit(database, g_state.granularity, variant)

        # First, we need to make our splits.
        make_splits(dataset)

        # Then, save our metadata right before we start training.
        record_meta()

        # This will be set to False in our SIGINT handler when the user requests
        #   we jump to the test phase. Training data is dumped to disk when the
        #   `do_train` function exists so we may continue later.
        g_state.keep_training = True

        # The user can request to keep training after do_test, so we loop here
        while g_state.keep_training:
            # Now, we start training.
            do_train()

            # And if we get here, we do test.
            g_state.keep_training = do_test()

if __name__ == "__main__":
    default_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    main()
