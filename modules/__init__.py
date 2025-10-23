"""Compatibility shim package.

The test suite expects a top‑level ``modules`` package mirroring the SDK
structure (e.g. ``modules.auth.auth_client``).  This shim simply re‑exports the
real implementation from ``wme_sdk`` without altering any functionality.
"""

