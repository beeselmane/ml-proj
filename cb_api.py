# -*- coding: utf-8 -*-

from enum import Enum

# For signing requests
import hashlib
import base64
import hmac
import time

# For sending requests
from requests.auth import AuthBase
import requests

# For sys.stderr
import sys

# General constants
ENDPOINT = 'api.pro.coinbase.com'
PROTOCOL = 'https'

# This class holds constants representing various API endpoints
class Endpoint:
    PRODUCTS = 'products'
    PROFILES = 'profiles'
    CANDLES = lambda p: 'products/{}/candles'.format(p)
    PRODUCT = lambda p: 'products/{}'.format(p)

# APIError class. This class represents various errors the API class may encounter.
# Some of these are actually deprecated... (I actually think I only end up using one of them...)
class APIError(Exception):
    Underlying = 0
    Authentication = 1
    Request = 3
    UnknownMethod = 3

    def __init__(self, cause = Underlying):
        self._cause = cause

    def __str__(self):
        return {
            APIError.Underlying : 'Underlying',
            APIError.Authentication : 'Authentication',
            APIError.Request : 'Request',
            APIError.UnknownMethod: 'UnknownMethod'
        }[self._cause]

# APIResponse Class. This class encapsulates the response from an API request
# Objects of this type are returned by methods in the API class
class APIResponse:
    def __init__(self, underlying):
        self._underlying = underlying

    def headers(self):
        return self._underlying.headers

    def content(self):
        return self._underlying.json()

    def status(self):
        return self._underlying.status_code

    def did_succeed(self):
        return (200 <= self._underlying.status_code) and (self._underlying.status_code < 300)

# Auth Class. This handles authentication by API key.
# Authentication must be done for many API calls.
# We extent AuthBase and implement __call__ to be allowed to be passed directly
#   as the `auth` argument of request() calls.
class Auth(AuthBase):
    HEADER_KEY  = 'CB-ACCESS-KEY'
    HEADER_PASS = 'CB-ACCESS-PASSPHRASE'
    HEADER_TIME = 'CB-ACCESS-TIMESTAMP'
    HEADER_SIGN = 'CB-ACCESS-SIGN'

    def __init__(self, key, passphrase, secret):
        self._passphrase = passphrase
        self._key = key

        # We expect the secret in base64 encoding.
        self._secret = base64.b64decode(secret)

    # 'Calling' an authentication object signs the provided request with the given key/pass/secret pair.
    # This adds the necessary headers to the request object provided
    def __call__(self, request):
        now = str(time.time())

        info = '{}{}{}{}'.format(now, request.method, request.path_url, (request.body or ''))

        signature = hmac.new(self._secret, info.encode('ascii'), hashlib.sha256).digest()

        request.headers.update({
            Auth.HEADER_KEY  : self._key,
            Auth.HEADER_PASS : self._passphrase,
            Auth.HEADER_TIME : now,
            Auth.HEADER_SIGN : base64.b64encode(signature).decode('utf-8')
        })

        return request

# API Class. This handles building and executing requests in the proper format.
class API:
    def __init__(self, auth, endpoint = ENDPOINT, protocol = PROTOCOL, validate = True, timeout = 30, verbose = False):
        self._request_url = f'{protocol}://{endpoint}'
        self._verbose = verbose
        self._auth = auth

        self._session = requests.Session()
        self._timeout = timeout

        if self._verbose:
            print(f'API Endpoint: {endpoint}', file = sys.stderr)
            print(f'API Protocol: {protocol}', file = sys.stderr)

            print(f'API URL: {self._request_url}', file = sys.stderr)

        if validate and not self.__validate_credentials():
            raise APIError(cause = APIError.Authentication)

    def __validate_credentials(self):
        return self.request(Endpoint.PROFILES).did_succeed()

    def __request_headers(self):
        return {
            'Accept' : 'application/json',
            'Content-Type' : 'application/json'
        }

    def request(self, endpoint, method = 'GET', params = None):
        request_url = '{}/{}'.format(self._request_url, endpoint)
        api_headers = self.__request_headers()

        if self._verbose:
            print(f'{method.upper()}: {request_url}', file = sys.stderr)

            if params:
                print(f'{method.upper()}: params={params}', file = sys.stderr)

        response = APIResponse(self._session.request(
            method = method,
            url = request_url,
            auth = self._auth,
            timeout = self._timeout,
            headers = api_headers,
            params = params
        ))

        if self._verbose:
            print(f'{method.upper()}: Status: {response.status()}', file = sys.stderr)

        return response

    # This could probably be optimized, but oh well...
    def batch(self, endpoints):
        return [self.request(e) for e in endpoints]

    def candles(self, product, start, end, granularity):
        params = {
            'start' : start.isoformat(),
            'end' : end.isoformat(),

            'granularity': granularity
        }

        return self.request(Endpoint.CANDLES(product), params = params)
