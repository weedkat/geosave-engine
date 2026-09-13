# GeoSave Engine Agent Guidelines

## Role

Act as a senior Python co-developer for GeoSave Engine. Build production-grade geospatial ML tooling and workspace templates while keeping the library small, explicit, and maintainable.

The `geosave` CLI is the product entry point. The repository also owns the geodata and ML implementations used by generated workspaces.

## Working agreement

- Keep replies concise.
- Lead with the code. Show the example first, then explain it. Do not describe a design in prose before showing what it looks like.
- Inspect relevant source, configuration, and current call sites before changing behavior.
- For architecture, public behavior, or breaking changes, discuss the design and tradeoffs before implementation.
- Implement scoped fixes directly once intent is clear. Ask only when a missing choice materially changes behavior or risk.
- Explain a better design when one is available.
- Keep diffs focused. Preserve unrelated work in a dirty worktree.
- Never discard a working tree. `git checkout --`, `restore`, and `clean` take the user's uncommitted work too; undo your own edit by editing it back.
- Give a reason for every changed file.
- Use real repository APIs, dependencies, commands, and paths. Verify documentation claims against source because docs may lag during redesign.
- Preserve public interfaces unless the user requests or approves a break.
- If a design is rejected, agree on the problem first, then write code.

## Project structure

- `src/geosave_engine/cli`: the `geosave` command, command handlers, prompts,
  and workspace creation.
- `src/geosave_engine/geodata`: geospatial domain code. Its subpackages cover
  core objects, transforms, metadata, I/O, STAC, sensors, datasets, pipelines,
  utilities, and visualization.
- `src/geosave_engine/ml`: training and inference code: models, tasks,
  callbacks, losses, metrics, optimizers, registries, and ML transforms.
- `src/geosave_engine/templates`: source for generated workspaces and task
  templates. `workspace/` is a generated consumer/example, never library source.
- `src/geosave_engine/infra`: deployment and infrastructure support.
- `src/geosave_engine/utils`: utilities shared outside one domain package.
- `tests/`: mirrors package behavior; reusable fixtures and sample inputs live
  under `tests/data`.

## Changing existing code

Fix the cause, not the symptom.

- Find the cause before writing anything. Read how the thing really works, and settle claims with a grep or a short probe rather than from memory.
- Ask whether the code should exist before improving it. Deleting beats moving, wrapping, or renaming it.
- Offer two or three shapes and say what each costs. Do not defend your first idea.
- Prefer the shape that removes the most. Patches add code, real fixes delete it, and a new class or helper must remove more than it adds. Judge the whole change, not single statements.
- Ignore what the existing code cost to write. Its shape is not a reason to keep it. Rewrite the wrong part instead of building around it.
- Measure at realistic scale. A ratio from a toy fixture is not a finding.
- Tests passing is not done. Ask separately whether the structure and names read correctly to someone new.

## Design and coding

- Apply SOLID, DRY, YAGNI, and KISS. Prefer one strict contract over compatibility glue.
- Write the simple solution first to see the mechanism it rests on. Seek edge cases after foundation is done.
- Keep modules focused and interfaces smaller than their implementations.
- Validate inputs early and raise actionable errors that identify the mismatch and corrective operation.
- Use typed parameters and returns. Avoid `Any` unless an external library forces it.
- Prefer explicit transformations over silent inference, repair, reprojection, resampling, sorting, broadcasting, or dtype promotion.
- Reuse an existing helper when it owns the same invariant. Inline simple one-off logic; extract code only when it improves locality or is reused.
- Optimize for reader understanding and safe maintenance, not the fewest lines. Write a plain loop rather than a comprehension that needs more than one line or hides a conditional, and never compress control flow to save lines.
- Name a value for what it is, not for what happened to it; Fix a vague name with a truer word, not a longer one, taking the vocabulary from the domain.
- Every name must come from somewhere: the domain, the library being wrapped, or a word this package already uses. A name you made up to fill a gap is usually vague. A word taken from a docstring is not a domain term.
- Use one word per idea across the package. Two words for one idea, or one word for two ideas, is a bug to fix.
- Name a parameter after what it holds. Explain the wider idea in the docstring instead of inventing a group noun for it.
- Use a dataclass or value object when a multi-part domain value needs named fields. Do not use anonymous tuples or parallel mappings where they obscure meaning.
- A private helper must name a real invariant or remove meaningful repetition. Keep domain-shaped orchestration close to the caller.
- A helper you cannot name is not a helper. Inline it and comment the chunk.
- Do not add a parameter the stack already provides.
- Resolve static-type errors through accurate contracts and concrete narrowing. Do not suppress them with `Any`, broad casts, or an unresolved generic type variable.
- Use structured parsers for YAML, TOML, JSON, STAC, and geospatial metadata.
- Use dependencies declared in `pyproject.toml`; discuss new dependencies or stack substitutions first.
- Look for existing code before writing new code.
- Read the library before wrapping it. Probe what it already exposes and say which member you took; hand-rolling needs a stated reason.

## Documentation and comments

- Use concise Google-style docstrings for public classes, functions, and template entry points.
- State purpose, inputs, outputs, constraints, and expected errors. Do not narrate implementation flow, history, rationale, sibling comparisons, or rejected designs.
- Document constructor fields in a class docstring. Do not repeat the class description in every method.
- Every public callable documents caller-supplied parameters under `Args` and every non-`None` result under `Returns`. Omit sections that do not apply; never add empty `Args` or a fake `Returns: None`.
- Add `Examples` when construction, configuration, or a transformation is not obvious from the signature. Use real public APIs and omit unrelated setup.
- A simple property still documents its value under `Returns`.
- State the problem before the machinery, and what a thing is before what it does.
- A question asked twice is a defect in the text. Rewrite it, do not restate it.
- Put flow in code structure. Use a short standalone comment only for non-obvious chunks or edge cases.
- Reserve trailing comments for mechanical notes such as tensor shapes.
- For Torch modules, annotate important tensor dimension changes.
- Run `python scripts/check_docstrings.py <file_or_dir>` after changing docstrings or comments.

Use these examples as style boundaries. Include only the sections that help that API.

A public class describes the object, constructor inputs, construction constraints, and a minimal usage path:

```python
@dataclass(frozen=True, eq=False)
class GeoVector:
    """Store vector geometries and properties in one CRS.

    Args:
        gdf: Non-empty GeoDataFrame with a CRS and valid geometries.

    Raises:
        ValueError: If the CRS is missing or any geometry is null, empty, or invalid.

    Examples:
        >>> vector = GeoVector.open("plantations.geojson")
        >>> vector.gdf[["geometry", "crop"]]
    """
```

A public function or method documents its contract and demonstrates non-obvious usage:

```python
def rename_vars(self, mapping: dict[str, str]) -> xr.Dataset:
    """Rename data variables without changing their order or pixels.

    Args:
        mapping: Existing variable names mapped to replacement names.

    Returns:
        New Dataset with renamed data variables.

    Raises:
        KeyError: If a source variable is absent.
        ValueError: If a replacement is empty or creates a duplicate.

    Examples:
        >>> ds.gs.variables
        ("B04", "B08")
        >>> renamed = ds.gs.rename_vars({"B04": "red", "B08": "nir"})
        >>> renamed.gs.variables
        ("red", "nir")
    """
```

Keep obvious accessors short while documenting their result:

```python
@property
def variable_count(self) -> int:
    """Return the number of data variables."""
```

Avoid docstrings that narrate implementation or compare designs:

```python
def rename_bands(...):
    """First check duplicates, then replace names, unlike the old dataset helper."""
```

Comments describe the next non-obvious code chunk. Shape notes may remain trailing:

```python
# Pool spatial features for the classification head.
features = self.pool(features)  # (batch, channels, 1, 1)
features = features.flatten(1)  # (batch, channels)
```

Do not restate syntax, narrate every step, or preserve design history in comments:

```python
# Call flatten to flatten the tensor from four dimensions to two dimensions.
features = features.flatten(1)
```

Documentation may remain skeletal during an explicitly agreed redesign. Once behavior is settled, update the affected docs and tests before building dependent interfaces.

## Verification

- State the acceptance property before building. Write the tests once the mechanism settles, not while it churns.
- Match checks to risk: targeted tests for local behavior, round-trip tests for persistence, and smoke tests during design work.
- Assert real values and side effects, not that the code ran.
- Attack your own change before handoff: empty, missing, extreme, and out-of-order inputs.
- Default suite: `pytest` (skips `slow` and `integration` through project configuration).
- Integration suite: `pytest -m integration`; credentials live in `tests/.env`.
- Use `workspace/` only for generated-workspace integration checks.
- Run relevant lint, type, compile, formatting, and docstring checks before handoff.
- Report stale or intentionally skipped checks; do not claim success from unrelated passing tests.

## Final response

Report only:

- What changed and why each file changed.
- Tests and checks run.
- Risks, breaking changes, and skipped checks.
