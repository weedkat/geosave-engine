import typer
import questionary as qu

beautiful_theme = qu.Style([
    ('qmark', 'fg:#61afef bold'),       # Color of the question mark
    ('question', 'bold'),               # Style of the question text
    ('answer', 'fg:#98c379 bold'),      # Color of the submitted answer text
    ('pointer', 'fg:#56b6c2 bold'),     # Color of the selection cursor/arrow
    ('highlighted', 'fg:#56b6c2 bold'), # Color of the currently hovered option
    ('selected', 'fg:#98c379'),         # Color of checked items (in checkboxes)
    ('separator', 'fg:#abb2bf'),        # Color of visual dividers
    ('instruction', 'fg:#5c6370 italic'), # Color of helper text (e.g. "(Use arrow keys)")
    ('text', 'fg:#abb2bf'),             # Color of standard typed text
    ('disabled', 'fg:#4b5263 italic')   # Color of disabled options
])

def prompt_required_text(message: str, default: str = "") -> str:
    answer = qu.text(
            message,
            validate=lambda text: bool(text.strip()) or "Cannot be empty.",
            style=beautiful_theme,
        ).ask()

    if answer is None:
        raise typer.Abort()

    return answer.strip()


def prompt_optional_text(message: str, default: str = "") -> str | None:
    """Prompt for optional text input. Returns None if the user submits an empty string."""
    answer = qu.text(
         message,
         style=beautiful_theme,
         ).ask()
    
    if answer is None:
        raise typer.Abort()

    return answer if answer.strip() else ''


def prompt_select(message: str, choices: list[str] | list[qu.Choice]) -> str:
    answer = qu.select(
        message,
        choices=choices,
        style=beautiful_theme,
    ).ask()

    if answer is None:
        raise typer.Abort()

    return answer