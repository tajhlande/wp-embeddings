"""Compatibility wrapper for :pyclass:`wme_sdk.auth.auth_client.AuthClient`.

The test suite imports ``modules.auth.auth_client.AuthClient``.  To avoid
duplicating code we simply re‑export the implementation from the real SDK.
"""

from wme_sdk.auth.auth_client import AuthClient as _AuthClient

# Export under the expected name
AuthClient = _AuthClient

