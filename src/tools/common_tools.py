"""Shared text helpers."""


def normalize_description(user_description: str) -> str:
    """Normalize a human description without an unnecessary LLM call."""
    return " ".join(user_description.strip().split())
