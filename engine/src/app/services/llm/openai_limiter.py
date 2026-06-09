"""Optional process-wide RPM token bucket for OpenAI calls.

Controlled by Settings.OPENAI_RPM_LIMIT (default 0 = disabled). When disabled,
acquire() is a no-op and there is zero behavior change vs the previous codebase.

When enabled (OPENAI_RPM_LIMIT > 0), each call to acquire() blocks until a slot
is available, preventing rate-limit 429 errors from the provider.

Usage:
    limiter = OpenAILimiter(settings.OPENAI_RPM_LIMIT)
    async with limiter.acquire():
        resp = await client.chat.completions.create(...)
"""

from __future__ import annotations

import contextlib
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

log = logging.getLogger(__name__)


class OpenAILimiter:
    """Process-wide RPM token bucket. No-op when rpm_limit <= 0.

    Args:
        rpm_limit: Max requests per minute. 0 or negative disables limiting.
    """

    def __init__(self, rpm_limit: int = 0) -> None:
        self._rpm_limit = rpm_limit
        self._limiter = None
        if rpm_limit > 0:
            try:
                from aiolimiter import AsyncLimiter

                self._limiter = AsyncLimiter(rpm_limit, 60.0)
                log.info("OpenAI RPM limiter enabled: %d rpm", rpm_limit)
            except ImportError:
                log.warning("aiolimiter not installed; OpenAI RPM limiter disabled")

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Acquire a rate-limit slot. No-op if limiter is disabled.

        Yields:
            None - use as async context manager around the API call.
        """
        if self._limiter is not None:
            async with self._limiter:
                yield
        else:
            with contextlib.nullcontext():
                yield
