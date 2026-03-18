"""Hashing utilities for privacy-preserving input fingerprinting."""

import hashlib

import xxhash


def hash_input(data: str, method: str = "xxhash") -> str:
    """Hash input data for privacy-preserving storage.

    Returns a prefixed hash string: 'xxh64:<hex>' or 'sha256:<hex>'.
    """
    if method == "xxhash":
        digest = xxhash.xxh64(data.encode()).hexdigest()
        return f"xxh64:{digest}"
    elif method == "sha256":
        digest = hashlib.sha256(data.encode()).hexdigest()
        return f"sha256:{digest}"
    else:
        raise ValueError(f"Unsupported hash method: {method}")


def hash_content(data: str) -> str:
    """Hash content using xxhash for fast fingerprinting."""
    digest = xxhash.xxh64(data.encode()).hexdigest()
    return f"xxh64:{digest}"
