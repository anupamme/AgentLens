"""Shared validation helpers for the AgentLens trace schema."""

import re

HASH_PATTERN = re.compile(r"^(xxh64|sha256):[a-f0-9]+$")


def is_valid_hash(value: str) -> bool:
    """Check if a string is a valid AgentLens hash format."""
    return bool(HASH_PATTERN.match(value))


def validate_non_empty_string(value: str, field_name: str) -> str:
    """Validate that a string is non-empty after stripping whitespace."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty or whitespace-only")
    return stripped
