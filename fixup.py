#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file implements a command line tool to fixup empty data gotten from the Coinbase API.
# It will read from a provided SQLite database and look for rows in a given Dataset (granularity)
#   with the `complete` column set to 0. It will interpolate from the prior/following entries.
# That is, `volume` will remain 0, but the other fields will be set to the following (for each
#   target that's missing data):
# *_close = *_min = following.*_open
# *_open  = *_max = prior.*_close
#
# This modification maintains the sanity of entries assigned 0's by the database
#   by interpolation, but keeps volume 0 (or whatever it was before fixup, the API is quite weird sometimes...)

# As a helpful tip, all of these commands can be run as such to apply to all granularities:
# t=(60 300 900 3600 21600 86400) printf '%s\n' "${t[@]}" | xargs -I {} ./fixup.py --db-loc=coinbase.sqlite --granularity={} ...

from data import Database

from cmd import open_dataset_or_exit
from cmd import make_granularity
from cmd import find_database

# For 'API_PRODUCTS'
import data

from datetime import timedelta
from datetime import datetime
from datetime import timezone

import pickle
import sys

################################################################################
# Constants/Globals

v = False

################################################################################
# Core Logic

# These are currently unimplemented, I didn't get any data that needed these methods...
def fixup_head(dataframe):
    return dataframe

def fixup_tail(dataframe):
    return dataframe

def do_fixups(database, granularity, do_verify, force):
    fixup_database = open_dataset_or_exit(database, granularity, Database.Dataset.VARIANT_PATCHED)
    dataset = open_dataset_or_exit(database, granularity, Database.Dataset.VARIANT_RAW)

    if fixup_database.current_range():
        if force:
            print('Warning: Fixed up dataset already exists. Will overwrite.', file = sys.stderr)

            fixup_database.clear()
        else:
            print('Error: Fixed up dataset already exists.', file = sys.stderr)
            print('Error: Refusing to overwrite...')

            sys.exit(1)

    if do_verify and not dataset.verify():
        print('Warning: Failed to verify the table for the requested granularity!', file = sys.stderr)
        print('Error: This is bad. You probably have to clone the data again...', file = sys.stderr)
        print('Note: I\'m not even going to try to fixup this database, bye.', file = sys.stderr)

        sys.exit(1)

    print('Info: Dataset verification passed.', file = sys.stderr)

    dataframe = dataset.select_all(only_complete_records = False)

    if dataframe.empty:
        print('Warning: This table appears to be empty!', file = sys.stderr)
        print('Note: Exiting here with successful exit code.', file = sys.stderr)

        sys.exit(0)

    if len(dataframe[dataframe['complete'] == False]) == 0:
        print('Info: No fixups appear to be necessary for this database.', file = sys.stdout)
        print('Info: Simply copying to the destination and exiting...', file = sys.stdout)

        fixup_database.save(dataframe)

        sys.exit(0)

    print(f'Info: Begin fixups for {len(dataframe[dataframe["complete"] == False])} rows...', file = sys.stderr)

    # We fix the head and tail of our dataframe first.
    # These require a special case, since we can't look
    #   directly at both sides of the data point temporily.
    if not dataframe.iloc[0].complete:
        dataframe = fixup_head(dataframe)

    if not dataframe.iloc[-1].complete:
        dataframe = fixup_tail(dataframe)

    rows_to_fix = dataframe.index[dataframe['complete'] == False]

    for i in rows_to_fix:
        time = dataframe.loc[i]['time']
        dataframe.loc[i] = dataframe.loc[i - 1]
        dataframe.loc[i, 'time'] = time

    print('Saving fixed up dataset...', end = '')

    fixup_database.save(dataframe)

    print(' Done')

def fixup_norm(database, granularity):
    norm_database = open_dataset_or_exit(database, granularity, Database.Dataset.VARIANT_NORM)
    dataset = open_dataset_or_exit(database, granularity, Database.Dataset.VARIANT_PATCHED)

    if norm_database.current_range():
        norm_database.clear()

    dataframe = dataset.select_all()

    # We modify here
    m_dataframe = dataframe[dataframe.columns[1:]]

    mean = m_dataframe.mean(axis = 0)
    deviation = m_dataframe.std(axis = 0)

    result = (m_dataframe.sub(mean, axis = 1)).div(deviation, axis = 1)

    # Save this without any modification
    result.insert(loc = 0, column = 'time', value = dataframe['time'])

    print('Saving normalized dataset...', end = '')

    norm_database.save(result)

    print(' Done')

    print('Saving metadata...', end = '')

    with open(f'md/candles_{granularity}s.pyc', 'wb') as fp:
        pickle.dump({
            'deviation' : deviation,
            'mean' : mean
        }, fp)

    print(' Done')

################################################################################
# Main Function

def main():
    # I only want to access this here.
    globals()['v'] = '-v' in sys.argv

    # Select a given candle granularity
    granularity = make_granularity(sys.argv)

    # Find database
    database_path = find_database(sys.argv)

    # Parse database file to use and open database
    with Database(database_path, verbose = v) as database:
        if '--show-info' in sys.argv:
            dataset = open_dataset_or_exit(database, granularity, Database.Dataset.VARIANT_RAW)

            __range = dataset.current_range()

            # We just dump table info and exit
            print(f'Info for dataset with granularity={granularity} in database \'{database_path}\':')
            print(f'Total entries: {len(dataset)}')
            print(f'Incomplete entries: {dataset.count_incomplete()}')

            if __range:
                print(f'Earliest record at: {__range[0].isoformat()}')
                print(f'Latest record at: {__range[1].isoformat()}')
            else:
                print('Missing range data.')

            pass
        else:
            # We perform fixups (note that this is NOT in place)
            if not ('--skip-base' in sys.argv):
                do_fixups(database, granularity, not ('--no-verify' in sys.argv), ('--force' in sys.argv))

            fixup_norm(database, granularity)

if __name__ == "__main__":
    main()
