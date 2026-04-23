from __future__ import annotations

import shlex


def normalize_slug(value: str) -> str:
    """Lower-case, collapse underscores/hyphens to spaces, squeeze whitespace."""
    return " ".join(value.replace("_", " ").replace("-", " ").lower().split())


def resolve_from_choices(raw_value: str, choices: list[str]) -> str | None:
    """Fuzzy-match `raw_value` against `choices` using slug normalization.

    Returns the original choice string on match, or None if no match.
    """
    normalized = normalize_slug(raw_value)
    norm_map = {normalize_slug(choice): choice for choice in choices}
    if normalized in norm_map:
        return norm_map[normalized]

    target_tokens = set(normalized.split())
    if not target_tokens:
        return None
    for choice in choices:
        choice_tokens = set(normalize_slug(choice).split())
        if target_tokens == choice_tokens:
            return choice
    return None


def parse_shell_args(text: str) -> list[str]:
    """Shell-aware split of a user-entered argument string."""
    stripped = text.strip()
    if not stripped:
        return []
    return shlex.split(stripped)
