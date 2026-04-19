from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class FakePrompter:
    """In-memory Prompter stub for unit tests.

    Queues are consumed in FIFO order per prompt kind. Tests set them up with
    the exact answers their flow should receive, then assert on the recorded
    prompt messages.
    """

    selects: list[str | None] = field(default_factory=list)
    select_mappings: list[str | None] = field(default_factory=list)
    texts: list[str | None] = field(default_factory=list)
    confirms: list[bool] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def select(
        self,
        message: str,
        choices: Sequence[str],
        *,
        default: str | None = None,
    ) -> str | None:
        self.calls.append(("select", message))
        return self.selects.pop(0)

    def select_mapping(
        self,
        message: str,
        choices: Sequence[tuple[str, str]],
        *,
        default: str | None = None,
    ) -> str | None:
        self.calls.append(("select_mapping", message))
        return self.select_mappings.pop(0)

    def text(self, message: str, *, default: str | None = None) -> str | None:
        self.calls.append(("text", message))
        return self.texts.pop(0)

    def confirm(self, message: str, *, default: bool = False) -> bool:
        self.calls.append(("confirm", message))
        return self.confirms.pop(0)


@dataclass
class StubConsole:
    """In-memory Console stub that records each message by level."""

    messages: list[tuple[str, str]] = field(default_factory=list)

    def info(self, msg: str) -> None:
        self.messages.append(("info", msg))

    def warn(self, msg: str) -> None:
        self.messages.append(("warn", msg))

    def error(self, msg: str) -> None:
        self.messages.append(("error", msg))

    def success(self, msg: str) -> None:
        self.messages.append(("success", msg))
