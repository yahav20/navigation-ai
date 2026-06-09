"""Shared HTTP session for all providers.

SSL certificate verification is disabled because the Windows system certificate
store is not wired into the Python/requests trust chain on this machine.
All traffic is outbound-only to known public APIs — no sensitive data is sent.
"""
from __future__ import annotations

import warnings

import requests
import urllib3

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

_session = requests.Session()
_session.verify = False


def get(url: str, **kwargs) -> requests.Response:
    """Drop-in for requests.get() using the shared SSL-disabled session."""
    kwargs.setdefault("timeout", 15)
    return _session.get(url, **kwargs)
