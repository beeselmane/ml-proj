#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data import Database

from cmd import make_granularity
from cmd import find_database
from cmd import date_string

import matplotlib
import sys

################################################################################
# Constants/Globals/Initialization Code

# We're going to make some pretty vector graphics I can include in my report.
matplotlib.use('Agg')

################################################################################
# Main Function

def main():
    # Select a given candle granularity
    granularity = make_granularity(sys.argv)

    # Find database
    database_path = find_database(sys.argv)

    # Parse database file to use and open database
    with Database(database_path) as database:
        yield

if __name__ == '__main__':
    main()
