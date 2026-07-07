"""Stable identifiers for retry-safe operations."""

import hashlib
import json
from typing import Any


def canonical_hash(value: Any) -> str:
    """Return a deterministic SHA-256 hash for JSON-compatible data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def operation_key(*parts: Any) -> str:
    """Build a compact deterministic key for one persistent side effect."""
    return canonical_hash(parts)
