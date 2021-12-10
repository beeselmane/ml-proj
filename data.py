# -*- coding: utf-8 -*-

from datetime import datetime
from datetime import timezone
from typing import NamedTuple
from itertools import chain

import sqlite3
import pandas
import numpy
import sys

################################################################################
# Constants

# We can maybe also include 'MUSD-EUR', but it doesn't seem to have as much history....
# Should we change to ETH-EUR?
API_PRODUCTS = ['ETH-BTC', 'BTC-USD', 'ETH-EUR']

# This is the dimension of input for the neural network we train
DATA_DIMENSION = 16

################################################################################
# Convienience Classes

# Database error!
class DatabaseError(Exception):
    def __str__(self):
        return 'Database Inconsistency!'

# META:
# - interval
class Datapoint(NamedTuple):
    # Tuple entry names
    time : int
    # Do we have all the other data points below?
    complete : bool

    eth_btc_min : float
    eth_btc_max : float
    eth_btc_open : float
    eth_btc_close : float
    eth_btc_vol : float

    btc_usd_min : float
    btc_usd_max : float
    btc_usd_open : float
    btc_usd_close : float
    btc_usd_vol : float

    eth_eur_min : float
    eth_eur_max : float
    eth_eur_open : float
    eth_eur_close : float
    eth_eur_vol : float

################################################################################
# Main Classes

# Database class. This class represents the database where we store data for our model.
# We wrap an sqlite connection and expose methods to get/set the data we want to expose
# Note: This class is not secure, and I can think of at least one possible injection attack.
#   Don't use it for anything more than it is.
class Database:
    # These are the valid time intervals supported by the Coinbase Pro API
    ACCEPTABLE_INTERVALS = [60, 300, 900, 3600, 21600, 86400]

    # Database.Dataset represents a given dataset table.
    # Each table for a given time interval is unique.
    class Dataset:
        # Acceptable dataset variants.
        VARIANT_PATCHED = '_patched'
        VARIANT_RAW = ''

        def __init__(self, database, interval, variant):
            self._database = database
            self._interval = interval

            # Who are we?
            self._name = f'candles_{self._interval}s{variant}'
            self._variant = variant

            # We check if the table exists here. See sqlite3 documentation for method
            table = self._database._transact('SELECT name FROM sqlite_master WHERE type = \'table\' AND name = ?', self._name)

            # Need to create table if this happens
            if len(table.fetchall()) == 0:
                complete_column_desc = 'complete INTEGER,' if self._variant == Database.Dataset.VARIANT_RAW else ''

                self._database._transact(f'''
                    CREATE TABLE {self._name} (
                        time          INTEGER PRIMARY KEY,
                        {complete_column_desc}

                        eth_btc_open  REAL,
                        eth_btc_close REAL,
                        eth_btc_max   REAL,
                        eth_btc_min   REAL,
                        eth_btc_vol   REAL,

                        btc_usd_open  REAL,
                        btc_usd_close REAL,
                        btc_usd_max   REAL,
                        btc_usd_min   REAL,
                        btc_usd_vol   REAL,

                        eth_eur_open  REAL,
                        eth_eur_close REAL,
                        eth_eur_max   REAL,
                        eth_eur_min   REAL,
                        eth_eur_vol   REAL
                    )
                ''')

            if self._database._verbose:
                print(f'Info: Opened table "{self._name}"', file = sys.stderr)

        # Do a loose verification of the dataset, looking for holes in data
        # Note: This is a bit of a stupid algorithm, but that helps to ensure it actually works properly.
        # Note: As a stupid algorithm, it takes forever. This is mainly because I have to execute all the
        #   queries below seperately. Well actually, I don't, but I do because of laziness mainly.
        def verify(self):
            # We don't need to verify patched datasets, they are 'sealed' and should always be valid.
            if self._variant == Database.Dataset.VARIANT_PATCHED:
                return True

            # We're going to look to make sure we have all of the times we expect to have by selecting them all.
            # This way, if we don't have some, the total number of rows returned will be less than expected.
            check_range = self.current_range()

            # We have nothing yet.
            if not check_range:
                if self._database._verbose:
                    print('Note: verify() called on empty dataset!', file = sys.stderr)

                return True

            total_seconds = int((check_range[1] - check_range[0]).total_seconds())

            if total_seconds % self._interval:
                print(f'Error: Dataset verification FAIL! ({total_seconds} % {self._interval} = {total_seconds % self._interval})', file = sys.stderr)

            query_string = f'''SELECT COUNT(1) FROM {self._name} WHERE time = ?;'''
            initial_time = int(check_range[0].timestamp())

            # I can't use executemany() to select :(
            # We expect a total count of times where a given expected time actually corresponds to a record
            row_count = sum(self._database._transact(query_string, t).fetchall()[0][0] for t in range(initial_time, initial_time + total_seconds, self._interval))

            # For now, we verify if we have all of the expected times in our range.
            return (row_count == (total_seconds / self._interval))

        # Create and return a new dataset which is a normalized copy of this dataset.
        # Return with it the original variance and mean of this dataset (and add to meta table)
        # Note that if a normalized copy of this dataset already exists, a new one will only be
        #   recomputed if recompute = True; otherwise, it will simply return the preexisting dataset.
        def normalized_copy(self, recompute = False):
            pass

        def insert_rows(self, points):
            # TODO: Check inserted types here.

            # TODO: Is there a better way to do this than all of these question marks?
            #   Probably, maybe something like `(['?'] * 17).join(', ')`, but whatever
            self._database._batch(f'''INSERT INTO {self._name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', points)

        def insert_single(self, point):
            self.insert_rows([point])

        # Save a dataframe to this dataset
        def save(self, dataframe):
            if self._variant != Database.Dataset.VARIANT_RAW:
                dataframe = dataframe.drop('complete', axis = 1, errors = 'ignore')

            dataframe.to_sql(self._name, self._database._connection, index = False, if_exists = 'append')

        def select_range(self, earliest, latest, only_complete_records = True, inclusive = True):
            mod = '=' if inclusive else ''

            select_target = f'(SELECT * FROM {self._name} WHERE complete = 1)' if only_complete_records and self._variant == Database.Dataset.VARIANT_RAW else f'{self._name}'
            query_string = f'SELECT * FROM {select_target} WHERE (time >= ? AND time <{mod} ?)'

            params = [int(earliest.timestamp()), int(latest.timestamp())]

            dataframe = pandas.read_sql_query(query_string, self._database._connection, params = params)

            if dataframe.empty:
                return dataframe

            if only_complete_records and self._variant == Database.Dataset.VARIANT_RAW:
                return dataframe.drop('complete', axis = 1)
            else:
                return dataframe

        def select_all(self, only_complete_records = True):
            return self.select_range(*self.current_range(), only_complete_records, inclusive = True)

        # THIS DROPS ALL DATA. USE WITH CAUTION.
        def clear(self):
            print(f'WARNING: Dropping all data from table \'{self._name}\'!!!', file = sys.stderr)

            self._database._transact(f'DELETE FROM {self._name}')

        def count_incomplete(self):
            return self._database._transact(f'''SELECT COUNT(1) FROM {self._name} WHERE complete = 0''').fetchall()[0][0]

        def __len__(self):
            return self._database._transact(f'''SELECT COUNT(1) FROM {self._name}''').fetchall()[0][0]

        def current_range(self):
            queries = [f'SELECT MIN(time) FROM {self._name}', f'SELECT MAX(time) FROM {self._name}']
            rows = [r[0] for r in chain.from_iterable(self._database._transact(q).fetchall() for q in queries)]

            # If we have max, we ought have min
            if len(rows) > 2 or len(rows) == 1:
                print(f'Error: Attempt to get current range for dataset \'{self._name}\' got {rows} rows!', file = sys.stderr)
                raise DatabaseError()

            if None in rows:
                return None

            return [datetime.fromtimestamp(r, tz = timezone.utc) for r in rows]

    def __init__(self, db_path, verbose = False):
        self._verbose = verbose
        self._path = db_path

    def __enter__(self):
        self._connection = sqlite3.connect(self._path)

        # This line will throw an exception if sqlite3.connect() fails above.
        self._connection.cursor().close()

        # We need to pass ourself back out here
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._connection.close()

    # This method handles committing automatically
    def _transact(self, command, *params):
        cursor = self._connection.cursor()

        cursor.execute(command, params)

        self._connection.commit()

        return cursor

    # This method batches transactions with a shared command string and an iterable of parameters
    def _batch(self, command, param_iterable):
        cursor = self._connection.cursor()

        cursor.executemany(command, param_iterable)

        self._connection.commit()

        return cursor

    # `variant` determines the level of fixups which have been performed.
    # The fetch script writes the VARIANT_RAW dataset, and the fixup script modifies
    #   it and writes the VARIANT_PATCHED variant
    def open_dataset(self, interval, variant):
        if not isinstance(interval, int):
            print(f'Error: Interval must be an integer! (found {type(interval)})', file = sys.stderr)
            return None

        if interval not in Database.ACCEPTABLE_INTERVALS:
            print(f'Error: Interval "{interval}" is not acceptable!', file = sys.stderr)
            return None

        return Database.Dataset(self, interval, variant)
