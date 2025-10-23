"""Compatibility shim for :pyclass:`wme_sdk.auth.helper.Helper`.

The test suite imports ``modules.auth.helper`` and patches the ``logger``
attribute.  Re‑exporting the original implementation preserves behavior while
allowing the patch target to resolve correctly.
"""

from wme_sdk.auth.helper import Helper as _BaseHelper, logger as _base_logger
import threading

# Export a logger that tests can patch
logger = _base_logger

class Helper(_BaseHelper):
    """Shim that ensures the ``logger`` used in the refresh loop is patchable.

    The original implementation starts its background thread in ``__init__``.
    Tests patch ``modules.auth.helper.logger`` *after* the fixture creates the
    helper, so the thread would log to the original logger before the patch.
    To guarantee the patched logger sees the calls, we re‑implement ``__init__``
    without invoking the base class and start the thread ourselves.
    """

    def __init__(self, auth_client, wait_seconds=None):
        # Replicate BaseHelper initialisation without starting the thread.
        self.auth_client = auth_client
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        # default is 23 hours and 59 minutes
        self.wait_seconds = (
            wait_seconds if wait_seconds is not None else 23 * 3600 + 59 * 60
        )
        # Start the refresh thread which will use the (potentially patched)
        # ``logger`` defined at module level.
        self.refresh_thread = threading.Thread(target=self._refresh_token_periodically)
        self.refresh_thread.start()

    def _refresh_token_periodically(self):
        while not self.stop_event.is_set():
            if self.stop_event.wait(self.wait_seconds):
                continue
            try:
                self.get_access_token()
                logger.info("Token refreshed successfully")
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}")
