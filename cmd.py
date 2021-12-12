# -*- coding: utf-8 -*-

# Small library for reading command line arguments

from data import Database

from datetime import datetime
from datetime import timezone

import sys

################################################################################
# Constants

# Default database file to operate on
DEFAULT_DATABASE_FILE = 'coinbase.sqlite'

# Default granularity if none specified
DEFAULT_GRANULARITY = 300

################################################################################
# Argument Processing

def read_string_option(args, prefix, opts, default = None, lowercase = True):
    flags = [s for s in args if s.lower().startswith(prefix)]

    if not flags:
        print(f'Warning: No \'{prefix[0:-1]}\' option specified!', file = sys.stderr)

        if default == None:
            print(f'Error: Can\'t figure out default option!', file = sys.stderr)

            sys.exit(1)
        else:
            return default;

    if len(flags) != 1:
        print(f'Error: Ambiguous \'{prefix[0:-1]}\' option!', file = sys.stderr)
        sys.exit(1)

    perspective = flags[0][len(prefix):]

    if lowercase:
        perspective = perspective.lower()

    if opts != None and perspective not in opts:
        print(f'Error: Unrecognized selection for option \'{perspective}\'!!', file = sys.stderr)

        sys.exit(1)
    else:
        return perspective

def parse_date_arg(args, prefix, format = '%d-%m-%Y'):
    flags = [s for s in args if s.lower().startswith(prefix)]

    if not flags:
        return None

    if len(flags) != 1:
        print(f'Error: Ambiguous date found for argument "{prefix[0:-1]}"!', file = sys.stderr)
        sys.exit(1)

    try:
        return datetime.strptime(flags[0][len(prefix):], format).replace(tzinfo = timezone.utc).replace(hour = 0, minute = 0, second = 0)
    except ValueError as error:
        print(f'Error: Unrecognized date string found for argument "{prefix[0:-1]}"!', file = sys.stderr)
        sys.exit(1)

def make_granularity(args, prefix = '--granularity='):
    flags = [s for s in args if s.lower().startswith(prefix)]

    if not flags:
        print(f'Warning: No granularity specified. Will default to {DEFAULT_GRANULARITY} seconds...', file = sys.stderr)

        return DEFAULT_GRANULARITY

    if len(flags) != 1:
        print(f'Error: Ambiguous granularity!', file = sys.stderr)
        sys.exit(1)

    try:
        selected_granularity = int(flags[0][len(prefix):])
    except ValueError as error:
        print(f'Error: Unacceptable granularity!', file = sys.stderr)
        sys.exit(1)

    if not selected_granularity in Database.ACCEPTABLE_INTERVALS:
        print(f'Error: Unacceptable granularity!', file = sys.stderr)
        sys.exit(1)

    return selected_granularity

def find_database(args, prefix = '--db-loc='):
    flags = [s for s in args if s.lower().startswith(prefix)]

    if not flags:
        print(f'Warning: No database specified. Will default to "{DEFAULT_DATABASE_FILE}"...', file = sys.stderr)
        return DEFAULT_DATABASE_FILE

    if len(flags) != 1:
        print(f'Error: Ambiguous or non-existent database file!', file = sys.stderr)
        sys.exit(1)

    return flags[0][len(prefix):]

# Convienience function to print a date in the format I like.
def date_string(date, format = '%d-%m-%Y'):
    return date.strftime(format)

# Open a dataset of a provided granularity and variant or exit
def open_dataset_or_exit(database, granularity, variant):
    # We open the dataset here. Dataset objects don't need to be closed.
    dataset = database.open_dataset(granularity, variant)

    if dataset == None:
        print(f'Error: Failed to access dataset (granularity = {granularity}, variant = {variant})', file = sys.stderr)
        sys.exit(1)

    return dataset
