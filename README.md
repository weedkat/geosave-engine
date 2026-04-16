# geosave-engine
A backend engine for remote sensing task, full pipeline from data gathering, training, and serving

## Development Workflow

Use `uv` to manage the project environment instead of `pip install -e .`.

```bash
uv sync --locked --no-editable
uv run geosave --help
```

If you want the development environment to reflect source changes without reinstalling, omit `--no-editable`.
