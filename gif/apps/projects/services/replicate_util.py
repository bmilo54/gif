"""Shared Replicate runner with spacing + 429 retry.

Free / low-credit accounts are limited to 6 predictions/min and burst 1.
Calling two models per card immediately 429s.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time

from django.conf import settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_LAST_MONO = 0.0
# 6/min with burst 1 → leave ~11s between creates.
_MIN_INTERVAL = 11.0


def client():
    token = getattr(settings, 'REPLICATE_API_TOKEN', '')
    if not token:
        return None
    try:
        import httpx
        import replicate
    except ImportError:
        return None
    os.environ['REPLICATE_API_TOKEN'] = token
    # Grounded SAM cold-starts past the library default (30s read / 60s wait).
    return replicate.Client(
        api_token=token,
        timeout=httpx.Timeout(30.0, connect=30.0, read=180.0, write=60.0),
    )


def _status(exc):
    return getattr(exc, 'status', None) or getattr(exc, 'status_code', None)


def _throttled(exc) -> bool:
    if _status(exc) == 429:
        return True
    text = str(exc).lower()
    return '429' in text or 'throttl' in text or 'rate limit' in text


def _not_found(exc) -> bool:
    if _status(exc) == 404:
        return True
    text = str(exc).lower()
    return '404' in text or 'could not be found' in text


def _timed_out(exc) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return 'timeout' in name or 'timed out' in text or 'time out' in text


def _empty_detection(exc) -> bool:
    text = str(exc).lower()
    return (
        '0 elements' in text
        or 'reshape tensor' in text
        or 'unspecified dimension' in text
        or 'no object' in text
    )


def _retry_wait(exc) -> float:
    match = re.search(r'resets in ~?(\d+)\s*s', str(exc), re.I)
    if match:
        return min(45.0, float(match.group(1)) + 1.5)
    return _MIN_INTERVAL


def _materialize(output):
    """Consume iterator models (Grounded SAM) so failures surface here."""
    if output is None:
        return None
    if isinstance(output, (bytes, bytearray, str, dict)):
        return output
    if hasattr(output, 'read') or hasattr(output, 'url'):
        return output
    try:
        return [item for item in output]
    except TypeError:
        return output


def run(model, payload, *, retries=5):
    """Create one prediction, waiting out the low-credit rate limit."""
    replicate = client()
    if not replicate or not model:
        return None
    global _LAST_MONO
    with _LOCK:
        for attempt in range(retries):
            gap = _MIN_INTERVAL - (time.monotonic() - _LAST_MONO)
            if gap > 0:
                logger.info('Waiting %.1fs before Replicate call (rate limit)', gap)
                time.sleep(gap)
            try:
                # wait=False: POST returns immediately, then poll. Prefer: wait
                # times out at 60s on Grounded SAM cold start.
                output = _materialize(replicate.run(model, input=payload, wait=False))
                _LAST_MONO = time.monotonic()
                return output
            except Exception as exc:
                _LAST_MONO = time.monotonic()
                if _not_found(exc):
                    logger.error(
                        'Replicate 404 for %s — use owner/name:version, not owner/name',
                        model,
                    )
                    return None
                if _empty_detection(exc):
                    logger.info(
                        'Replicate %s found no objects (%s)',
                        model, str(exc).split('\n', 1)[0][:160],
                    )
                    return None
                retryable = (_throttled(exc) or _timed_out(exc)) and attempt < retries - 1
                if retryable:
                    wait = _retry_wait(exc) if _throttled(exc) else 8.0
                    kind = '429' if _throttled(exc) else 'timeout'
                    logger.warning(
                        'Replicate %s on %s, retry %d/%d in %.1fs',
                        kind, model, attempt + 1, retries, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.warning('Replicate run failed (%s): %s', model, exc)
                return None
    return None
