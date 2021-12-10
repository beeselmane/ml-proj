# This file describes the API access to the coinbase API.
# Don't put code here. This is a scratch file.

###########################################################################################################
# Request headers:
#
# CB-ACCESS-KEY: Our API key
# CB-ACCESS-PASSPHRASE: Our API passphrase (cache this for a given API key)
#
# CB-ACCESS-TIMESTAMP: Time our request was generated (must be within 30 seconds of request receipt)
# - This is in UNIX timestamp format (seconds since 00:00:00 UTC, 1 Jan 1970)
#
# CB-ACCESS-SIGN: SHA-256-HMAC signature in base64
# - Hash function should be run on `timestamp +            method +     request path + body`
#                                   ^ CB-ACCESS-TIMESTAMP  ^ uppercase  ^ /*/*         ^ raw request body
###########################################################################################################


#################################################
# We need to limit to a given number of reqs/sec
# Look into this later....
#################################################


###############################################################################
# Timestamps returned in ISO 8601 with microseconds.
#
# We can parse these with dateutil.parser.parse(timestamp)
# This returns a datetime.datetime object
#
# Decimal numbers are quoted, we can use python's decimal library for figures
#  (or multiply to get all of the divisions of each bitcoin/eth right)
#
# Identifiers are 128-bit UUIDs. Python also has a library for these.
###############################################################################


###########################
# Page-id headers for provided response:
#
# CB-BEFORE
# CB-AFTER
#
# before,after,limit
###########################


###########################
# 200 - Success
# 2** - Also ok
#
# 400 - Bad format
# 401 - Unauthorized
# 403 - Forbidden
# 404 - Not Found
#
# 500 - Internal Error
###########################
