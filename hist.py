#!/usr/bin/env python3

from data import Datapoint
from data import Database

import cb_api as api
import data

from datetime import timedelta
from datetime import datetime
from datetime import timezone
from datetime import date

import json
import time
import sys
import os

################################################################################
# Constants

DEFAULT_DATABASE_FILE = 'coinbase.sqlite'
DEFAULT_AUTH_FILE = 'coinbase.cred'

DEFAULT_GRANULARITY = 300

# Used to calculate fetch bucket size
SEC_PER_DAY = 60 * 60 * 24

# The API enforces this as the max number of datapoints we can copy per request
REQ_MAX_CANDLES = 200

# This is our single global. I do the UNIX command tool thing here, if you don't like it oh well...
v = False

################################################################################
# Argument Processing

# You would think python would have some equivalent to getopt()
# And I assume it doesn, but I didn't use it, because this is not a professional
#   tool, and I'm lazy. There, now you know.

def make_auth(args, prefix = '--auth-file='):
    flags = [s for s in args if s.lower().startswith(prefix)]

    if not flags:
        print(f'Warning: No auth file specified. Will default to looking at "{DEFAULT_AUTH_FILE}"...', file = sys.stderr)
        flags = [prefix + DEFAULT_AUTH_FILE]

    if len(flags) != 1:
        print('Error: Ambiguous auth file!', file = sys.stderr)
        sys.exit(1)

    path = flags[0][len(prefix):]

    if v:
        print(f'Info: Using auth file: {path}', file = sys.stderr)

    with open(path, 'rb') as file:
        creds = json.load(file)

        return api.Auth(creds['key'], creds['passphrase'], creds['secret'])

def parse_date_arg(args, prefix, format = '%d-%m-%Y'):
    flags = [s for s in args if s.lower().startswith(prefix)]

    if not flags:
        return None

    if len(flags) != 1:
        print(f'Error: Ambiguous date found for argument "{prefix[0:-1]}"!', file = sys.stderr)
        sys.exit(1)

    try:
        return datetime.strptime(flags[0][len(prefix):], format)
    except ValueError as error:
        print(f'Error: Unrecognized date string found for argument "{prefix[0:-1]}"!', file = sys.stderr)
        sys.exit(1)

# Convienience function to print a date in the format I like.
def date_string(date, format = '%d-%m-%Y'):
    return date.strftime(format)

def make_granularity(args, prefix = '--granularity='):
    flags = [s for s in args if s.lower().startswith(prefix)]

    if not flags:
        print(f'Warning: No granularity specified. Will default to {DEFAULT_GRANULARITY} seconds...')

        return DEFAULT_GRANULARITY

    if len(flags) != 1:
        print(f'Error: Ambiguous granularity!')
        sys.exit(1)

    try:
        selected_granularity = int(flags[0][len(prefix):])
    except ValueError as error:
        print(f'Error: Unacceptable granularity!')
        sys.exit(1)

    if not selected_granularity in Database.ACCEPTABLE_INTERVALS:
        print(f'Error: Unacceptable granularity!')
        sys.exit(1)

    return selected_granularity

def find_database(args, prefix = '--db-loc='):
    flags = [s for s in args if s.lower().startswith(prefix)]

    if not flags:
        print(f'Warning: No database specified. Will default to "{DEFAULT_DATABASE_FILE}"...')
        return DEFAULT_DATABASE_FILE

    if len(flags) != 1:
        print(f'Error: Ambiguous or non-existent database file!')
        sys.exit(1)

    return flags[0][len(prefix):]

################################################################################
# Core Logic

def do_copy_targets(database, cbapi, targets):
    # Fetch info about targets we want to update.
    # We use this to ensure the targets are actually valid
    endpoints = [api.Endpoint.PRODUCT(p) for p in targets]
    target_info = cbapi.batch(endpoints)

    if sum([not t.did_succeed() for t in target_info]):
        print(f'Error: Failed to get target info for targets in {targets}!', file = sys.stderr)
        sys.exit(1)

    # Unwrap our target info dictionaries
    target_info = [t.content() for t in target_info]

    print(f'Info: (Target Info) {target_info}', file = sys.stderr)

    # TODO: Insert into database

# Convienience function to process a batch of data
def batch_process():
    yield

def do_clone(database, cbapi, start, end, granularity, targets):
    # Convienience function to iterate over a date range
    def date_range(start, end, inc = 1):
        for i in range(0, (end - start).days, inc):
            yield start + timedelta(days = i)

    # Find the greatest multiple k of n such that k <= m
    def greatest_multiple(n, m):
        for k in range(m):
            if (n * (k + 1)) > m:
                return k

        return m

    # We open the dataset here. Dataset objects don't need to be closed.
    dataset = database.open_dataset(granularity)

    if not dataset:
        print('Error: Failed to access dataset')
        sys.exit(1)

    # TODO: Check earliest/latest values in dataset, clamp start/end date so we don't duplicate data.

    print(f'Begin fetch candles...')

    # Number of candles per day (per target)
    # Note that granularity is necessarily evenly divisible by SEC_PER_DAY
    candles_per_day = SEC_PER_DAY / granularity

    if candles_per_day <= REQ_MAX_CANDLES:
        increment = greatest_multiple(candles_per_day, REQ_MAX_CANDLES)

        for first_day in date_range(start, end, inc = increment):
            last_day = first_day + timedelta(days = increment)

            # Note that the Coinbase API fetches data including the last requested time,
            #   so we request until last_day - granularity
            last_req_datetime = datetime(year = last_day.year, month = last_day.month, day = last_day.day) - timedelta(seconds = granularity)

            # If we went past the end, clamp back to the proper end date.
            if end < last_day:
                last_req_datetime = end
                last_day = end

            print(f'{date_string(first_day)} : {date_string(last_day)}...', end = '')
            sys.stdout.flush()

            # Actually fetch the data from the API
            api_candles = [cbapi.candles(t, first_day, last_req_datetime, granularity) for t in targets]
            candles = [c.content() for c in api_candles]

            if sum([not c.did_succeed() for c in api_candles]) != 0:
                print(' [failed]')
                print('Warning: Failed to fetch API data for one or more targets!', file = sys.stderr)

                # We skip this record if it fails to download
                continue

            # Currently, we just skip datapoint where we don't have full data.
            if len(set([len(c) for c in candles])) != 1:
                print(' [failed]')
                print(f'Warning: Downloaded data lengths don\'t match! (lengths = {[len(c) for c in candles]})', file = sys.stderr)

                # Again, for now, we just skip this.
                continue

            # We're going to fill this below.
            new_batch = []

            # Candle data is cloned in a set format of [unixtime, low, high, open, close, volume]
            # We want to zip the candle data together from each of our targets, and put them all NamedTuple
            #   instances (in our Datapoint class). This is the format we use to insert into the Dataset instance.
            for point in zip(*candles):
                if len(set([p[0] for p in point])) != 1:
                    print(' [failed]')
                    print(f'Warning: Downloaded data has mismatched timestamps!', file = sys.stderr)

                    # I'm too lazy to fix this.
                    continue

                # This is very dependent on the format of Datapoint and API_PRODUCTS (which is the same as `targets` here).
                # There is probably a better way to do this, hiding the implementation details entirely except in data.py,
                #   but this works for the scope of what's currently implemented.

                # Note that I remove decimal points to store everything as an int.
                # The original numbers can be easily retrieved by multiplying by 100000000, 100, and 100 respectively.
                # Volume is simply stored as a float.
                # timestamp = int(point[0][0])
                # eth_btc = [int(p * 100000000) for p in point[0][1:5]] + [float(point[0][5])]
                # btc_usd = [int(p * 100) for p in point[1][1:5]] + [float(point[1][5])]
                # eth_eur = [int(p * 100) for p in point[2][1:5]] + [float(point[2][5])]

                new_batch += [Datapoint(
                    int(point[0][0]), # timestamp, we make sure they're all equivalent above
                    *point[0][1:], # ETH-BTC data
                    *point[1][1:], # BTC-USD data
                    *point[2][1:], # ETH-EUR data
                    True # complete, this is always true here.
                )]

            # Save the data we just gathered
            dataset.insert_rows(new_batch)

            # We can only make 10 requests per second.
            # While technically we could do mulitple loops per second,
            #   I'm okay with this delay, since I can just fetch in the background on my laptop.
            time.sleep(1)

            print(' [done]')
        # fetch multiple days; find maximum k : (candles_per_day * k) < REQ_MAX_CANDLES, pick that for increment
        # bucket count = 1
    else:
        # Split each day into buckets;
        pass

    # Fix bucket size per fetch (given that we can fetch at most 200 candles per request)
    #daily_entries = nn

    # I made date_range behave like range(). That is, it's exclusive for the final day.
    # for day in date_range(start.replace(day = start.day + 1), end):
    #     print(f'{date_string(day)}...', end = '')
    #     sys.stdout.flush()
    #
    #     #
    #
    #     time.sleep(1)
    #
    #     # SKELETON:
    #     # 1. Find split # so we have at most 200 datapoints / day
    #     # 2. Fetch them, coalesce until we have one full day.
    #     # 3. Insert into database, one day at a time.
    #
    #     print(' [done]')

#    candles = cbapi.candles('ETH-BTC', start, end, 900)

    #if candles.did_succeed():
    #    print(f'Candles for {start.isoformat()} to {end.isoformat()}: {json.dumps(candles.content(), indent = 4, sort_keys = True)}')

    #    candles = cbapi.request(api.Endpoint.CANDLES(target_info['id']))

################################################################################
# Main Function

def main():
    # I only want to access this here.
    globals()['v'] = '-v' in sys.argv

    # Load credentials from file if requested
    auth = make_auth(sys.argv)

    # Flag for dumping all available targets
    # We just exit after dumping if this is specified
    if '--dump-targets' in sys.argv:
        try:
            cbapi = api.API(auth, verbose = v)

            all_products = cbapi.request(api.Endpoint.PRODUCTS)

            if all_products.did_succeed():
                print(json.dumps(all_products.content(), indent = 4, sort_keys = True))
                sys.exit(0)
            else:
                print(f'Failed to retrieve product info! (status = {all_products.status()})', file = sys.stderr)
                sys.exit(1)
        except api.APIError as error:
            print('Caught error when attempting to dump targets!', file = sys.stderr)
            print(f'Error info: {error}', file = sys.stderr)

    start_date = parse_date_arg(sys.argv, '--from-date=')
    end_date = parse_date_arg(sys.argv, '--to-date=')

    # This is used in the defaults calculation below
    today = date.today()

    if not start_date:
        start_date = today.replace(month = 1, day = 1, year = today.year - 2)

        print(f'Warning: No start date specified! Starting at "{date_string(start_date)}"...', file = sys.stderr)

    if not end_date:
        end_date = today

        print(f'Warning: No end date specified! Ending on "{date_string(end_date)}"...', file = sys.stderr)

    print(f'Info: Will fetch data for {(end_date - start_date).days} days...', file = sys.stderr)

    # Select a given candle granularity
    granularity = make_granularity(sys.argv)

    # We let our data module tell us what to get
    targets = data.API_PRODUCTS

    # Parse database file to use and open database
    with Database(find_database(sys.argv)) as database:
        try:
            cbapi = api.API(auth, verbose = v)

            do_copy_targets(database, cbapi, targets)

            do_clone(database, cbapi, start_date, end_date, granularity, targets)
        except api.APIError as error:
            print('API Error!', file = sys.stderr)
            print(error, file = sys.stderr)

if __name__ == "__main__":
    main()
