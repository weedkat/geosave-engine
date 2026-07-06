from __future__ import annotations

from typing import Callable, TypeVar

import numpy as np

PipelineFn = Callable[..., "np.ndarray | dict[str, np.ndarray]"]

DERIVE_FUNCTIONS: dict[str, PipelineFn] = {}

T = TypeVar("T", bound=PipelineFn)


def derive_step(name: str) -> Callable[[T], T]:
    """Register a function for use as a ``pipeline:`` step's ``action:``.

    Never touches a ``GeoTile`` directly — the engine handles all tile
    mechanics generically, the function only ever sees plain arrays/scalars.
    Exactly two shapes exist. The star's position is the one tell — no other
    flag needed to know which one a function is:

    - **Reduce** (many bands in, one result out — the common case: ndvi,
      cloud masks, intersect/union, morphology...). Signature is always
      ``(*bands: np.ndarray, stac_item=None, resolution=None, **own_params) ->
      np.ndarray``. Band count/order is positional and fixed by what the
      function expects — spelled out in the function's own docstring so the
      yaml author knows how to order it. yaml ``bands:`` is a plain ordered
      list per upstream, matching that order. Fixed arity unpacks by tuple
      assignment (``nir, red = bands``) — Python itself raises a clear error
      on a count mismatch, no manual check needed. Variable/large arity
      (e.g. 10 s2cloudless bands, N masks to intersect) checks count
      explicitly since a plain ``np.stack``/loop wouldn't otherwise notice a
      wrong count. Returns a single plain ``np.ndarray`` — the engine names
      it after the yaml *step's* own name.
    - **Map** (bands in, same-shaped bands out, names must survive the round
      trip — only ``apply_scale`` so far). Signature is always ``(*,
      stac_item=None, resolution=None, **own_params, **bands: np.ndarray) ->
      dict[str, np.ndarray]``. Bands come in as ``**bands`` instead of
      ``*bands`` because the names need to come back out. Returns
      ``dict[str, np.ndarray]`` with those same keys.

    Every function also always declares ``stac_item`` and ``resolution`` as
    keyword-only params, even when unused — the engine always injects both,
    unconditionally, for every action. No signature inspection, no
    ``**kwargs`` sink to swallow them quietly: if a function ignores them,
    that's visible right in its signature. yaml ``params:`` can still
    override either by name. Anything more specific than these two generic
    tile-derived values (e.g. sun azimuth) is the function's own job to dig
    out of ``stac_item``.

    Everything else keyword-only comes from yaml ``params:``.

    Args:
        name: Key referenced by a step's ``action:`` field.

    Raises:
        ValueError: If ``name`` is already registered.

    Examples:
        Reduce — fixed arity, order-bound, tuple-unpacked::

            @derive_step("ndvi")
            def compute_ndvi(
                *bands: np.ndarray,
                stac_item: pystac.Item | None = None, resolution: float | None = None,
                eps: float = 1e-6,
            ) -> np.ndarray:
                nir, red = bands   # order: nir, red — raises if not exactly 2
                return (nir - red) / (nir + red + eps)

        .. code-block:: yaml

            pipeline:
              vegetation_index:              # -> output band is "vegetation_index"
                action: ndvi
                bands: {sentinel_2_l1c: [B08, B04]}   # order: nir, red
                params: {eps: 1.0e-6}

        Map — unknown/variable band count, each one transformed but still
        individually named on the way out::

            @derive_step("apply_scale")
            def apply_scale(
                *, stac_item: pystac.Item | None = None, resolution: float | None = None,
                mode="from_stac", scale=None, offset=0.0, **bands: np.ndarray,
            ) -> dict[str, np.ndarray]:
                ...
                return {name: arr * scale + offset for name, arr in bands.items()}

        .. code-block:: yaml

            pipeline:
              sentinel_2_l1c:
                action: apply_scale
                from: sentinel_2_l1c_dn   # every band on this tile gets scaled, names kept
                params: {mode: from_stac}
    """
    def decorator(fn: T) -> T:
        if name in DERIVE_FUNCTIONS:
            raise ValueError(f"Derive function {name!r} already registered")
        DERIVE_FUNCTIONS[name] = fn
        return fn
    return decorator
