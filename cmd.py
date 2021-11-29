# -*- coding: utf-8 -*-

# Small library for reading command line arguments

from data import Database

################################################################################
# Constants

# Default database file to operate on
DEFAULT_DATABASE_FILE = 'coinbase.sqlite'

# Default granularity if none specified
DEFAULT_GRANULARITY = 300

################################################################################
# Argument Processing

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
