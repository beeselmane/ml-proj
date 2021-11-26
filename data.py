from datetime import datetime
from typing import NamedTuple

import sqlite3
import pandas
import numpy
import sys

# We can maybe also include 'MUSD-EUR', but it doesn't seem to have as much history....
# Should we change to ETH-EUR?
API_PRODUCTS = ['ETH-BTC', 'BTC-USD', 'ETH-EUR']

# META:
# - interval
class Datapoint(NamedTuple):
    # Tuple entry names
    time : datetime

    eth_btc_open : int
    eth_btc_close : int
    eth_btc_max : int
    eth_btc_min : int
    eth_btc_vol : float

    btc_usd_open : int
    btc_usd_close : int
    btc_usd_max : int
    btc_usd_min : int
    btc_usd_vol : float

    eth_eur_open : int
    eth_eur_close : int
    eth_eur_max : int
    eth_eur_min : int
    eth_eur_vol : float

    # Do we have all the data points above?
    complete : bool

# Dataset class. This class represents the specific dataset we use for storing trade data.
# We wrap an sqlite connection and expose methods to get/set the data we want to expose
# Note: This class is not secure, and I can think of at least one possible injection attack.
#   Don't use it for anything more than it is.
class Dataset:
    # These are the valid time intervals supported by the Coinbase Pro API
    ACCEPTABLE_INTERVALS = [60, 300, 900, 3600, 21600, 86400]

    # Dataset.Table represents a given table in the dataset.
    # Each table for a given time interval is unique.
    class Table:
        def __init__(self, dataset, interval):
            self._interval = interval
            self._dataset = dataset

            # Who are we?
            self._name = f'candles_{self._interval}s'

            # We check if the table exists here. See sqlite3 documentation for method
            table = self._dataset.__transact("SELECT name FROM sqlite_master WHERE type = 'table' AND name = '?'", self._name)

            # Need to create table if this happens
            if len(table.fetchall()) is 0:
                self._dataset.__transact('''
                    CREATE TABLE ? (
                        time          INTEGER PRIMARY KEY,
                        complete      INTEGER,

                        eth_btc_open  INTEGER,
                        eth_btc_close INTEGER,
                        eth_btc_max   INTEGER,
                        eth_btc_min   INTEGER,
                        eth_btc_vol   REAL,

                        btc_usd_open  INTEGER,
                        btc_usd_close INTEGER,
                        btc_usd_max   INTEGER,
                        btc_usd_min   INTEGER,
                        btc_usd_vol   REAL,

                        eth_eur_open  INTEGER,
                        eth_eur_close INTEGER,
                        eth_eur_max   INTEGER,
                        eth_eur_min   INTEGER,
                        eth_eur_vol   REAL
                    )
                ''', self._name)

        def insert_rows(self, points):
            # TODO: Check type

            '''INSERT INTO ? (?)'''

        def insert_single(self, point):
            self.insert_rows(point)

        def select_range(self, earliest, latest, only_complete_records = True):
            query_string = 'SELECT * FROM ? WHERE time BETWEEN ? AND ?'

            if only_complete_records:
                query_string += ' WHERE complete = 1'

            params = [self._name, int(earliest.timestamp()), int(latest.timestamp())

            dataframe = pandas.read_sql_query(query_string, self._dataset._connection, params = params)

            if only_complete_records:
                return dataframe.drop('complete', 1)
            else
                return dataframe

        def select_all(self, only_complete_records = True):
            query_string = 'SELECT * FROM ?'

            if only_complete_records:
                query_string += ' WHERE complete = 1'

            return pandas.read_sql_query(query_string, self._dataset._connection, paramas = [self._name])

    def __init__(self, db_path):
        self._path = db_path

    def __enter__(self):
        self._connection = sqlite3.connect(self._path)

        # This line will throw an exception if sqlite3.connect() fails above.
        self._connection.cursor().close()

    def __exit(self, exc_type, exc_value, traceback):
        self._connection.close()

    # This method handles committing automatically
    def __transact(self, commands, *params):
        cursor = self._connection.cursor()

        cursor.execute(command, params)

        self._connection.commit()

        return cursor

    # Same as __transact, but with multiple commands
    def __batch(self, commands):
        cursor = self._connection.cursor()

        [cursor.execute(c) for c in commands]

        self._connection.commit()

        return cursor

    def open_table(self, interval):
        if interval is not int:
            print(f'Interval must be an integer! (found {type(interval})', file = sys.stderr)

        if interval not in Dataset.ACCEPTABLE_INTERVALS:
            print(f'Interval "{interval}" is not acceptable!', file = sys.stderr)

        return Table(self, interval)

