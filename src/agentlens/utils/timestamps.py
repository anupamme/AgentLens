"""UTC timestamp utilities."""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc)


def parse_utc(timestamp_str: str) -> datetime:
    """Parse an ISO 8601 timestamp string to a UTC datetime."""
    dt = datetime.fromisoformat(timestamp_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
