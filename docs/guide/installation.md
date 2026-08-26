# Installation

> **Status:** Development setup is current. Published release instructions will be restored after the release workflow is reverified.

## Development setup

Clone the repository, then install the locked environment:

```bash
uv sync
```

Run the CLI from the environment:

```bash
uv run geosave --help
```

Run the default test selection:

```bash
uv run pytest
```

Python 3.12 or newer is required. See `pyproject.toml` for current dependencies and commands.
