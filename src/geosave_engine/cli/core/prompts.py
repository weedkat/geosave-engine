import typer
import questionary as qu

PROMPT_STYLE = qu.Style(
    [
        ("qmark", "fg:#61afef bold"),
        ("question", "bold"),
        ("answer", "fg:#98c379 bold"),
        ("pointer", "fg:#56b6c2 bold"),
        ("highlighted", "fg:#56b6c2 bold"),
        ("selected", "fg:#98c379"),
        ("separator", "fg:#abb2bf"),
        ("instruction", "fg:#5c6370 italic"),
        ("text", "fg:#abb2bf"),
        ("disabled", "fg:#4b5263 italic"),
    ]
)


def prompt_required_text(message: str) -> str:
    """Ask for text that cannot be left empty.

    Args:
        message: Question shown to the user.

    Returns:
        The stripped answer.

    Raises:
        typer.Abort: The user cancelled the prompt.
    """
    answer = qu.text(
        message,
        validate=lambda text: bool(text.strip()) or "Cannot be empty.",
        style=PROMPT_STYLE,
    ).ask()

    if answer is None:
        raise typer.Abort()

    return answer.strip()


def prompt_optional_text(message: str) -> str:
    """Ask for text the user may leave empty.

    Args:
        message: Question shown to the user.

    Returns:
        The stripped answer, empty when the user submitted nothing.

    Raises:
        typer.Abort: The user cancelled the prompt.
    """
    answer = qu.text(message, style=PROMPT_STYLE).ask()

    if answer is None:
        raise typer.Abort()

    return answer.strip()


def prompt_select(message: str, choices: list[str] | list[qu.Choice]) -> str:
    """Ask the user to pick one of `choices`.

    Args:
        message: Question shown to the user.
        choices: Selectable values, or `questionary.Choice` entries.

    Returns:
        The selected value.

    Raises:
        typer.Abort: The user cancelled the prompt.
    """
    answer = qu.select(
        message,
        choices=choices,
        style=PROMPT_STYLE,
    ).ask()

    if answer is None:
        raise typer.Abort()

    return answer
