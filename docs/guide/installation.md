# Installation

Two channels, matching the two release paths in
`.github/workflows/release.yml`: a stable channel cut from version tags, and
a rolling dev channel rebuilt on every push to `main`.

## Stable release

Tagged `vX.Y.Z` pushes publish to PyPI:

```bash
pip install geosave-engine
# or
uv add geosave-engine
```

Pulls the latest tagged version — no flags needed.

## Rolling dev build

Every push to `main` builds a dev version (`<base>.dev<run_number>`, e.g.
`0.1.0.dev42`) and publishes it to TestPyPI, plus attaches the wheel to a
rolling GitHub Release tagged `dev`.

**From TestPyPI:**

```bash
pip install --pre --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ geosave-engine
```

`--pre` is required — dev versions are pre-releases, pip hides them by
default. `--extra-index-url` is required too — TestPyPI doesn't mirror
dependencies, so without it resolution fails on things like `torchgeo`.

**From GitHub, stable URL (recommended):** the `dev` release always carries
a fixed-name alias wheel alongside the real versioned one, so this URL never
changes even though the underlying build does:

```bash
pip install https://github.com/weedkat/geosave-engine/releases/download/dev/geosave_engine-0+latest-py3-none-any.whl
```

**From GitHub, one exact build:** the real filename bakes in the dev
version, so pin a specific run via `gh`:

```bash
gh release download dev --repo weedkat/geosave-engine --pattern 'geosave_engine-*.dev*.whl'
pip install ./geosave_engine-*.whl
```

## Working on GeoSave Engine itself

Clone the repo and use [uv](https://docs.astral.sh/uv/) directly — not a
`pip install`, this is for developing the library, not consuming it:

```bash
git clone git@github.com:weedkat/geosave-engine.git
cd geosave-engine
uv sync
uv run geosave --help
```
