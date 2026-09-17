"""
Shared utilities: logging configuration, disk-caching decorator, and
HTTP retry-with-exponential-backoff decorator.

All modules in this project import `setup_logging` and use the returned
logger rather than `print`.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(
    name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
) -> logging.Logger:
    """Return a named logger that writes to stdout with a consistent format.

    Args:
        name: Logger name — use ``__name__`` from the calling module.
        level: Minimum log level (default ``logging.INFO``).
        fmt: Log record format string.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def retry(
    max_attempts: int = 5,
    backoff_factor: float = 2.0,
    initial_wait: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator that retries a function with exponential backoff on failure.

    Args:
        max_attempts: Maximum number of total attempts (including the first).
        backoff_factor: Multiplicative factor applied to the wait time after
            each failure.
        initial_wait: Seconds to wait before the first retry.
        exceptions: Tuple of exception types that trigger a retry.

    Returns:
        Decorated callable.

    Example:
        >>> @retry(max_attempts=3, exceptions=(requests.HTTPError,))
        ... def fetch(url):
        ...     ...
    """
    _log = setup_logging(__name__)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wait = initial_wait
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        _log.error(
                            "Function %s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    _log.warning(
                        "Attempt %d/%d for %s failed (%s). Retrying in %.1fs …",
                        attempt,
                        max_attempts,
                        func.__name__,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    wait *= backoff_factor

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------


def disk_cache(cache_dir: Path, ignore_args: bool = False) -> Callable:
    """Decorator that persists function results to disk using pickle.

    The cache key is a SHA-256 hash of the function name plus its positional
    and keyword arguments (serialised with :func:`pickle`).  Subsequent calls
    with identical arguments load the cached result instead of recomputing.

    Args:
        cache_dir: Directory where ``.pkl`` cache files will be stored.
            Created automatically if it does not exist.
        ignore_args: If ``True`` the cache key depends only on the function
            name, so the first result is always reused regardless of arguments.

    Returns:
        Decorated callable.

    Example:
        >>> CACHE = Path("data/raw/.cache")
        >>> @disk_cache(CACHE)
        ... def expensive_request(url: str) -> dict: ...
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _log = setup_logging(__name__)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if ignore_args:
                raw = func.__name__.encode()
            else:
                try:
                    raw = pickle.dumps((func.__name__, args, kwargs))
                except Exception:
                    raw = func.__name__.encode()

            key = hashlib.sha256(raw).hexdigest()[:16]
            cache_file = cache_dir / f"{func.__name__}_{key}.pkl"

            if cache_file.exists():
                _log.debug("Cache hit → %s", cache_file.name)
                with cache_file.open("rb") as fh:
                    return pickle.load(fh)

            result = func(*args, **kwargs)
            with cache_file.open("wb") as fh:
                pickle.dump(result, fh)
            _log.debug("Cache saved → %s", cache_file.name)
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def throttle(calls_per_second: float = 10.0) -> Callable:
    """Decorator that rate-limits a function to ``calls_per_second``.

    Args:
        calls_per_second: Maximum call rate.  Default 10 respects EDGAR's
            recommended limit.

    Returns:
        Decorated callable.
    """
    min_interval = 1.0 / calls_per_second
    last_called: list[float] = [0.0]

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            elapsed = time.monotonic() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_called[0] = time.monotonic()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def safe_divide(numerator: float, denominator: float, default: float = float("nan")) -> float:
    """Divide two numbers, returning ``default`` instead of raising on zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when *denominator* is zero or NaN.

    Returns:
        ``numerator / denominator`` or *default*.
    """
    if denominator == 0 or denominator != denominator:  # NaN check
        return default
    return numerator / denominator
