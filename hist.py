#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file implements a command line tool to download candle data from the Coinbase Pro API.
# It's the only documentation I plan on making for itself, but it should be pretty self-explanatory,
#   and the defaults it picks have been deemed by myself to be sane.
# Note that you are expected to put a JSON file with your Coinbase Pro API keys somewhere and pass it
#   with the argument --auth-file=<file>

from data import Datapoint
from data import Database

from cmd import make_granularity
from cmd import parse_date_arg
from cmd import find_database
from cmd import date_string

import cb_api as api
import data

from datetime import timedelta
from datetime import datetime
from datetime import timezone
from datetime import date

from itertools import chain

import json
import time
import sys
import os

################################################################################
# Constants/Globals

# Default path to look for Coinbase Pro API keyfile
DEFAULT_AUTH_FILE = 'coinbase.cred'

# Used to calculate fetch bucket size
SEC_PER_DAY = 60 * 60 * 24

# The API enforces this as the max number of datapoints we can copy per request
REQ_MAX_CANDLES = 200

MAX_REQ_PER_SEC = 10

# Number of data points per candle.
# Currently, each candle is [max, min, open, close, volume]
CANDLE_POINTS = 5

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

################################################################################
# Core Logic

def do_copy_targets(database, cbapi, targets):
    # Fetch info about targets we want to update.
    # We use this to ensure the targets are actually valid
    endpoints = [api.Endpoint.PRODUCT(p) for p in targets]
    target_info = cbapi.batch(endpoints)

    if sum(not t.did_succeed() for t in target_info):
        print(f'Error: Failed to get target info for targets in {targets}!', file = sys.stderr)
        sys.exit(1)

    # Unwrap our target info dictionaries
    target_info = [t.content() for t in target_info]

    #print(f'Info: (Target Info) {target_info}', file = sys.stderr)

    # TODO: Insert into database

def do_clone(database, cbapi, start, end, granularity, targets):
    # Convienience function to iterate over a date range
    def date_range(start, end, inc = 1):
        for i in range(0, (end - start).days, inc):
            yield start + timedelta(days = i)

    # Find the greatest integer k such that (n * k) <= m
    def greatest_multiple(n, m):
        for k in range(m):
            if (n * (k + 1)) > m:
                return k

        return m

    def check_candles(api_candles):
        candles = [c.content() for c in api_candles]

        if sum(not c.did_succeed() for c in api_candles) != 0:
            return None

        return candles

    def batch_insert(dataset, candles, first_time, candles_per_batch):
        # We're going to fill this below.
        new_batch = []

        # Track where we are in each array. Note that the API seems to respond backward,
        #   so these should start at the end of each corresponding list
        # Need - 1 since python doesn't have -- prefix operator.
        indicies = [(len(c) - 1) for c in candles]

        # Candle data is cloned in a set format of [unixtime, low, high, open, close, volume]
        # We need to ensure in this loop that we actually have data for all given times.
        # Don't loop through the data we got, loop through the data we expected to get so we can
        #   zero-fill on data we didn't get that we expected to get.
        # Note that the API seems to respond backward for some reason...
        for t in range(candles_per_batch):
            expected_time = first_time + timedelta(seconds = (t * granularity))
            unixtime = int(expected_time.timestamp())

            # This is somewhat dependent on the format of Datapoint and API_PRODUCTS (which is the same as `targets` here).
            # There is probably a better way to do this, hiding the implementation details entirely except in data.py,
            #   but this works for the scope of what's currently implemented.
            # We want to merge all of the data we copied into NamedTuple instances (in our Datapoint class).
            # This is the format we use to insert into the Dataset instance.
            datapoint_list = [0] * (CANDLE_POINTS * len(candles))
            complete = True

            for i, target_candle in enumerate(candles):
                if target_candle[indicies[i]][0] == unixtime:
                    datapoint_list[(i * CANDLE_POINTS):((i + 1) * CANDLE_POINTS)] = target_candle[indicies[i]][1:]

                    # Why the fuck does this language not just let me do --
                    # the line above should have indicies[--i], clearly
                    indicies[i] -= 1
                else:
                    # We're missing at least one datapoint here.
                    complete = False

            # Build the next batch to insert
            new_batch += [Datapoint(unixtime, complete, *datapoint_list)]

        # We should use all the data we have available...
        if sum(indicies) != -len(candles):
            print(' [failed]')
            print(f'Warning: Data for batch pull not fully consumed ({sum(indicies) + len(candles)} points remaining)!', file = sys.stderr)

            # We can't really do much here, this is an internal error.
            return

        # Save the data we just gathered
        dataset.insert_rows(new_batch)

    # This function is the main beef behind the outer function.
    # We expect there to be no data in `database` in the range [start, end]
    def fetch_batch(dataset, start, end):
        print(f'Info: Fetching data from {date_string(start)} to {date_string(end)}...', file = sys.stderr)

        # Number of candles per day (per target)
        # Note that granularity is necessarily evenly divisible by SEC_PER_DAY
        candles_per_day = int(SEC_PER_DAY / granularity)

        if candles_per_day <= REQ_MAX_CANDLES:
            increment = greatest_multiple(candles_per_day, REQ_MAX_CANDLES)

            # We have batches of multiple days here
            candles_per_batch = int(increment * candles_per_day)

            for first_day in date_range(start, end, inc = increment):
                last_day = first_day + timedelta(days = increment)

                # Note that the Coinbase API fetches data including the last requested time,
                #   so we request until last_day - granularity
                last_req_datetime = last_day - timedelta(seconds = granularity)

                # If we went past the end, clamp back to the proper end date.
                if end < last_day:
                    first_req_datetime = first_day
                    last_req_datetime = end - timedelta(seconds = granularity)
                    last_day = end

                    # This is the last pass through the loop, we can update this here...
                    batch_seconds = (last_req_datetime - first_req_datetime).total_seconds()
                    candles_per_batch = int(batch_seconds / granularity) + 1

                print(f'{date_string(first_day)} : {date_string(last_day)}...', end = '')
                sys.stdout.flush()

                # Actually fetch the data from the API
                api_candles = [cbapi.candles(t, first_day, last_req_datetime, granularity) for t in targets]
                candles = check_candles(api_candles)

                # check_candles returns None on failure
                if not candles:
                    print(' [failed]')
                    print('Warning: Failed to fetch API data for one or more targets!', file = sys.stderr)

                    # We skip this record if it fails to download
                    continue

                # Convienience functions to insert a new batch of data
                batch_insert(dataset, candles, first_day, candles_per_batch)

                # We can only make 10 requests per second.
                # While technically we could do mulitple loops per second,
                #   I'm okay with this delay, since I can just fetch in the background on my laptop.
                time.sleep(1)

                print(' [done]')
        else:
            # We want as few batches as possible, but we want a whole number of batches (this simplifies the fetch loop below)
            batches_per_day = int(min([(candles_per_day / k) for k in range(1, REQ_MAX_CANDLES) if not (candles_per_day % k)]))
            candles_per_batch = int(candles_per_day / batches_per_day)

            # This is the amount of time elapsed per batch
            batch_timedelta = timedelta(seconds = int(candles_per_batch * granularity))

            for first_day in date_range(start, end):
                print(f'{date_string(first_day)}...', end = '')
                sys.stdout.flush() # stdout is normally flushed on newline

                if (batches_per_day * len(targets)) <= MAX_REQ_PER_SEC:
                    # We can fetch all at once (build a list of queries and run them)
                    start_times = [first_day + (i * batch_timedelta) for i in range(batches_per_day)]

                    # Coinbase API fetches data inclusive to the last time passed; ensure we don't fetch the start of the next batch
                    end_times = [t + (batch_timedelta - timedelta(seconds = granularity)) for t in start_times]

                    # This is a bit of a shit show. Basically, we make requests for each target, check their validity, and
                    #   chain them into single daily lists if they are valid. We end up making (batches_per_day * len(targets))
                    #   requests here, and we end up with lists of at most `candles_per_day` candles. This represents all of the
                    #   available data for a single day.
                    request_params = [[(p, *t, granularity) for t in zip(start_times, end_times)] for p in targets]
                    api_candles = [[cbapi.candles(*r) for r in l] for l in request_params]

                    if sum(sum(not c.did_succeed for c in l) for l in api_candles) != 0:
                        print(' [failed]')
                        print('Warning: Failed to fetch API data for one or more targets!', file = sys.stderr)

                        # We skip the entire day if this happens.
                        continue

                    # Note here again that the API returns data in reverse order, so we need to chain reversed lists (and reverse it again for `batch_insert`)
                    # Python doesn't seem to natively have anything that can help me here, I need to convert to list, reverse, chain, and list again
                    # It seems only list() can really be reversed, but when reversed, it returns a reversed_list iterator. Why normal iterators can't be reversed I don't know.
                    # Why reverse(list(*)) doesn't return an object of type list, I similarly do not know. This works, so I guess just don't touch it...
                    candles = [list(chain.from_iterable(reversed(list(c.content() for c in l)))) for l in api_candles]

                    # Insert data one day at a time
                    batch_insert(dataset, candles, first_day, candles_per_day)

                    # See above comment, we can only do 10 requests per second
                    time.sleep(1)
                else:
                    # We need to do multiple fetches per day :/
                    # I'm going to do this the dumb way...
                    # I hope you have two hours...

                    for i in range(batches_per_day):
                        start_time = first_day + (i * batch_timedelta)

                        # Same as above; don't want to clone the last candle twice.
                        end_time = start_time + batch_timedelta - timedelta(seconds = granularity)

                        # Do the fetch here
                        api_candles = [cbapi.candles(p, start_time, end_time, granularity) for p in targets]
                        candles = check_candles(api_candles)

                        if not candles:
                            print(' [failed]')
                            print('Warning: Failed to fetch API data for one or more targets!', file = sys.stderr)

                            # We skip this record if it fails to download
                            continue

                        # TODO: Actually, this might be an issue if we attempt to resume through a day?
                        #   Review the range comparison code and make sure this is okay...
                        # Just insert here, it's easier and probably not any slower
                        batch_insert(dataset, candles, start_time, candles_per_batch)

                        # Same as before
                        time.sleep(1)

                # Common '[done]' text here
                print(f' [done]')

    # We open the dataset here. Dataset objects don't need to be closed.
    dataset = database.open_dataset(granularity, Database.Dataset.VARIANT_RAW)

    if not dataset:
        print('Error: Failed to access dataset', file = sys.stderr)
        sys.exit(1)

    # We don't need to clone data we already have in our dataset.
    dataset_range = dataset.current_range()

    fetch_jobs = [[start, end]]

    # If we have data in [start, end], we're going to want to cut it out of the fetch.
    if dataset_range and (dataset_range[1] >= start or dataset_range[0] <= end):
        print(f'Info: Database already includes data from {dataset_range[0].isoformat()} to {dataset_range[1].isoformat()}...', file = sys.stderr)

        if dataset_range[1] >= end and start >= dataset_range[0]:
            print(f'Info: Requested data already exists in the requested database!', file = sys.stderr)
            sys.exit(0)

        # clamp fetch_jobs to whatever data we don't have.
        if dataset_range[0] <= start:
            fetch_jobs = [[dataset_range[1] + timedelta(seconds = granularity), end]]
        elif dataset_range[0] > start and dataset_range[1] < end:
            fetch_jobs = [
                [start, dataset_range[0] - timedelta(seconds = granularity)],
                [dataset_range[1] + timedelta(seconds = granularity), end]
            ]
        else: # dataset_range[1] >= end
            fetch_jobs = [[start, dataset_range[0] - timedelta(seconds = granularity)]]

    day_count = sum((f[1] - f[0]).days for f in fetch_jobs)

    print(f'Info: Will fetch data for {day_count} days...', file = sys.stderr)

    print(f'Begin fetch candles...')

    # Oh no I make a list I don't need here, 怎麼辦
    # Wow look I can put UTF-8 in my python source...
    [fetch_batch(dataset, *f) for f in fetch_jobs]

    print(f'Info: Finished fetching requested data.', file = sys.stderr)

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
    today = datetime.today().replace(tzinfo = timezone.utc)

    if not start_date:
        start_date = today.replace(day = 1, month = 1, year = today.year - 2)

        print(f'Warning: No start date specified! Starting at "{date_string(start_date)}"...', file = sys.stderr)

    if not end_date:
        end_date = today

        print(f'Warning: No end date specified! Ending on "{date_string(end_date)}"...', file = sys.stderr)

    if start_date >= end_date:
        print(f'Error: End date can\'t be prior to start date!', file = sys.stderr)
        sys.exit(1)

    # Select a given candle granularity
    granularity = make_granularity(sys.argv)

    # We let our data module tell us what to get
    targets = data.API_PRODUCTS

    # Parse database file to use and open database
    with Database(find_database(sys.argv), verbose = v) as database:
        try:
            cbapi = api.API(auth, verbose = v)

            do_copy_targets(database, cbapi, targets)

            do_clone(database, cbapi, start_date, end_date, granularity, targets)
        except api.APIError as error:
            print('API Error!', file = sys.stderr)
            print(error, file = sys.stderr)

if __name__ == "__main__":
    main()
