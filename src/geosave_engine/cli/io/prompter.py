from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import questionary


@runtime_checkable
class Prompter(Protocol):
    def select(
        self,
        message: str,
        choices: Sequence[str],
        *,
        default: str | None = None,
    ) -> str | None: ...

    def select_mapping(
        self,
        message: str,
        choices: Sequence[tuple[str, str]],
        *,
        default: str | None = None,
    ) -> str | None:
        """Choose among `(label, value)` pairs; returns the value or None."""

    def text(self, message: str, *, default: str | None = None) -> str | None: ...

    def confirm(self, message: str, *, default: bool = False) -> bool: ...


class QuestionaryPrompter:
    def select(
        self,
        message: str,
        choices: Sequence[str],
        *,
        default: str | None = None,
    ) -> str | None:
        return questionary.select(message, choices=list(choices), default=default).ask()

    def select_mapping(
        self,
        message: str,
        choices: Sequence[tuple[str, str]],
        *,
        default: str | None = None,
    ) -> str | None:
        q_choices = [questionary.Choice(label, value=value) for label, value in choices]
        return questionary.select(message, choices=q_choices, default=default).ask()

    def text(self, message: str, *, default: str | None = None) -> str | None:
        return questionary.text(message, default=default or "").ask()

    def confirm(self, message: str, *, default: bool = False) -> bool:
        return bool(questionary.confirm(message, default=default).ask())
