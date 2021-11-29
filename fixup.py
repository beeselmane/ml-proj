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
#   by interpolation, but keeps volume 0 (and complete = 0)

from data import Database

# For 'API_PRODUCTS'
import data

from datetime import timedelta
from datetime import datetime
from datetime import timezone

import sys

################################################################################
# Constants/Globals

v = False

################################################################################
# Main Function

def main():
    # I only want to access this here.
    globals()['v'] = '-v' in sys.argv

    sys.exit(0)

if __name__ == "__main__":
    main()
