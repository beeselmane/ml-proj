#!/usr/bin/env python3

import cb_api as api
import data

from datetime import datetime
from datetime import timezone

import json
import sys

DEFAULT_AUTH_FILE = 'coinbase.cred'

v = False

def make_auth(args, prefix = '--auth-file='):
    flags = [s for s in args if s.lower().startswith(prefix)]

    if not flags:
        print(f'Warning: No auth file specified. Will look for default at "{DEFAULT_AUTH_FILE}"', file = sys.stderr)
        flags = [prefix + DEFAULT_AUTH_FILE]

    if len(flags) != 1:
        print('Error: Ambiguous auth file!', file = sys.stderr)
        return None

    path = flags[0][len(prefix):]

    if v:
        print(f'Using auth file: {path}', file = sys.stderr)

    with open(path, 'rb') as file:
        creds = json.load(file)

        return api.Auth(creds['key'], creds['passphrase'], creds['secret'])

#def make_target(args, prefix = '--target='):
#    flags = [s for s in args if s.lower().startswith(prefix)]
#
#    if not flags:
#        return None
#
#    if len(flags) != 1:
#        print('Error: Ambigous target!')
#        return None
#
#    return flags[0].upper().removeprefix(prefix.upper())

def date_string(s):
    pass

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
            print(f'Error info: {error}')

    # We let our data module tell us what to get
    targets = data.API_PRODUCTS

    try:
        cbapi = api.API(auth, verbose = v)

        # Fetch info about targets we want to update.
        # We use this to ensure the targets are actually valid
        endpoints = [api.Endpoint.PRODUCT(p) for p in targets]
        target_info = cbapi.batch(endpoints)

        if sum([not t.did_succeed() for t in target_info]):
            print(f'Error: Failed to get target info for targets in {targets}!', file = sys.stderr)
            sys.exit(1)

        # Unwrap our target info dictionaries
        target_info = [t.content() for t in target_info]

        print(f'Targets: {target_info}')

        time = datetime.now().replace(hour = 0, minute = 0, second = 0)

        end = time.replace(year = time.year - 2)
        start = end.replace(day = end.day - 1)

        candles = cbapi.candles('ETH-BTC', start, end, 900)

        if candles.did_succeed():
            print(f'Candles for {start.isoformat()} to {end.isoformat()}: {json.dumps(candles.content(), indent = 4, sort_keys = True)}')

#        candles = cbapi.request(api.Endpoint.CANDLES(target_info['id']))
    except api.APIError as error:
        print('API Error!')
        print(error)

    return

if __name__ == "__main__":
    main()

