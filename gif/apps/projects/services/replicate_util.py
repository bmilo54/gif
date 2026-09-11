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
        import replicate
    except ImportError:
        return None
    os.environ['REPLICATE_API_TOKEN'] = token
    return replicate


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


def _retry_wait(exc) -> float:
    match = re.search(r'resets in ~?(\d+)\s*s', str(exc), re.I)
    if match:
        return min(45.0, float(match.group(1)) + 1.5)
    return _MIN_INTERVAL


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
                output = replicate.run(model, input=payload)
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
                if _throttled(exc) and attempt < retries - 1:
                    wait = _retry_wait(exc)
                    logger.warning(
                        'Replicate 429 on %s, retry %d/%d in %.1fs',
                        model, attempt + 1, retries, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.exception('Replicate run failed (%s)', model)
                return None
    return None
