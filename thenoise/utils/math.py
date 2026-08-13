"""Small, dependency-free math helpers shared across the codebase."""

from __future__ import annotations


def round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the nearest multiple of ``multiple``."""
    return ((value + multiple - 1) // multiple) * multiple
