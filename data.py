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

    # Do we have all the data points above?
    complete : bool

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
        def __init__(self, database, interval):
            self._database = database
            self._interval = interval

            # Who are we?
            self._name = f'candles_{self._interval}s'

            # We check if the table exists here. See sqlite3 documentation for method
            table = self._database._transact('SELECT name FROM sqlite_master WHERE type = \'table\' AND name = ?', self._name)

            # Need to create table if this happens
            if len(table.fetchall()) == 0:
                self._database._transact(f'''
                    CREATE TABLE {self._name} (
                        time          INTEGER PRIMARY KEY,
                        complete      INTEGER,

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

            print(f'Info: Opened table "{self._name}"', file = sys.stderr)

        def insert_rows(self, points):
            # TODO: Check inserted types here.

            # TODO: Is there a better way to do this than all of these question marks?
            self._database._batch(f'''INSERT INTO {self._name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', points)

        def insert_single(self, point):
            self.insert_rows([point])

        def select_range(self, earliest, latest, only_complete_records = True):
            query_string = 'SELECT * FROM ? WHERE time BETWEEN ? AND ?'

            if only_complete_records:
                query_string += ' WHERE complete = 1'

            params = [self._name, int(earliest.timestamp()), int(latest.timestamp())]

            dataframe = pandas.read_sql_query(query_string, self._database._connection, params = params)

            if only_complete_records:
                return dataframe.drop('complete', 1)
            else:
                return dataframe

        def select_all(self, only_complete_records = True):
            query_string = 'SELECT * FROM ?'

            if only_complete_records:
                query_string += ' WHERE complete = 1'

            return pandas.read_sql_query(query_string, self._database._connection, paramas = [self._name])

        def current_range(self):
            queries = [f'SELECT MIN(time) FROM {self._name}', f'SELECT MAX(time) FROM {self._name}']
            rows = [r[0] for r in chain.from_iterable(self._database._transact(q).fetchall() for q in queries)]

            # If we have max, we ought have min
            if len(rows) > 2 or len(rows) == 1:
                print(f'Error: Attempt to get current range for dataset \'{self._name}\' got {rows} rows!')
                raise DatabaseError()

            if None in rows:
                return None

            return [datetime.fromtimestamp(r, tz = timezone.utc) for r in rows]

    def __init__(self, db_path):
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

    def open_dataset(self, interval):
        if not isinstance(interval, int):
            print(f'Error: Interval must be an integer! (found {type(interval)})', file = sys.stderr)
            return None

        if interval not in Database.ACCEPTABLE_INTERVALS:
            print(f'Error: Interval "{interval}" is not acceptable!', file = sys.stderr)
            return None

        return Database.Dataset(self, interval)
