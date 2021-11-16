import sqlite3

class Datapoint:
    def fields():
        return {
            'time'          : 'text',
            'interval'      : 'integer',
            'volume'        : 'real',
            'eth_btc_open'  : 'integer',
            'eth_btc_close' : 'integer',
            'eth_btc_max'   : 'integer',
            'eth_btc_min'   : 'integer',
            'btc_usd_open'  : 'integer',
            'btc_usd_close' : 'integer',
            'btc_usd_max'   : 'integer',
            'btc_usd_min'   : 'integer',
            'usd_eur_open'  : 'integer',
            'usd_eur_close' : 'integer',
            'usd_eur_max'   : 'integer',
            'usd_eur_min'   : 'integer',
            'btc_scale'     : 'integer',
            'usd_scale'     : 'integer'
        }

    def __init__(self, time, interval, volume, eth_btc, btc_usd, eth_usd, usd_eur):
        self.interval = interval
        self.volume = volume
        self.time = time

        self.eth_btc = eth_btc
        self.btc_usd = btc_usd
        self.eth_usd = eth_usd
        self.usd_eur = usd_eur

    def flatten(self):
        return self.interval, self.volume, self.eth_btc.flatten(), self.btc_usd.flatten(),

class Candle:
    def __init__(self, open, close, min, max, interval):
        self.interval = interval

        self.open = open
        self.close = close
        self.min = min
        self.max = max

# Dataset class. This class represents the specific dataset we use for storing trade data.
# We wrap an sqlite connection and expose methods to get/set the data we want to expose
class Dataset:
    def __init__(self, db_path):
        self._path = db_path

    def __enter__(self):
        self._connection = sqlite3.connect(self._path)

        # This line will throw an exception if sqlite3.connect() fails above.
        self._connection.cursor().close()

    def __exit(self, exc_type, exc_value, traceback):
        self._connection.close()

    def __transact(self, commands):
        cursor = self._connection.cursor()

        [cursor.execute(c) for c in commands]

        cursor.commit()

    def open_table(self, interval):
        '''create table data_{} (
        )'''.format(interval)

    def insert_data()

