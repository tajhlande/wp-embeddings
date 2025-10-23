"""Shim package exposing the real SDK auth implementation.

The original library resides under ``wme_sdk.auth``.  Tests import from the
``modules`` namespace, so we forward the symbols here.
"""

from wme_sdk.auth.auth_client import AuthClient  # noqa: F401
from wme_sdk.auth.helper import Helper  # noqa: F401

