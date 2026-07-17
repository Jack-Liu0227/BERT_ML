"""Retry policy helpers for long-running Figure 4 API experiments."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable


_QUOTA_RESET_PATTERN = re.compile(
    r"Daily usage limit reached.*?Resets at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC",
    re.IGNORECASE,
)


def next_quota_reset(
    messages: Iterable[str],
    *,
    now: datetime | None = None,
) -> datetime | None:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    resets: list[datetime] = []
    for message in messages:
        for match in _QUOTA_RESET_PATTERN.finditer(str(message)):
            resets.append(
                datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            )
    future = [reset for reset in resets if reset > current]
    return max(future) if future else None
