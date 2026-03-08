"""
Utility functions for llmtest.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def retry_async(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Async retry decorator with exponential backoff.

    Usage:
        @retry_async(max_retries=3)
        async def call_llm():
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            delay = delay_seconds
            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


class Timer:
    """
    Context manager for timing operations.

    Usage:
        with Timer() as t:
            await some_operation()
        print(f"Took {t.elapsed_ms}ms")
    """

    def __init__(self):
        self._start: float = 0
        self._end: float = 0

    def __enter__(self) -> Timer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end = time.monotonic()

    @property
    def elapsed_ms(self) -> float:
        end = self._end or time.monotonic()
        return (end - self._start) * 1000

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ms / 1000


def truncate(text: str, max_length: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_cost(usd: float) -> str:
    """Format a cost value in USD."""
    if usd < 0.01:
        return f"${usd:.6f}"
    elif usd < 1.0:
        return f"${usd:.4f}"
    else:
        return f"${usd:.2f}"


def format_tokens(count: int) -> str:
    """Format token count with K/M suffixes."""
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate without a tokenizer.
    Rule of thumb: ~4 chars per token for English text.
    """
    return max(1, len(text) // 4)
