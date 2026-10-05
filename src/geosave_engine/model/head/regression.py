from __future__ import annotations

from geosave_engine.model.registry import register_model

from .dense import DenseHead


@register_model("head", "regression")
class RegressionHead(DenseHead):
    """Map one decoded feature map to per-pixel values.

    Args:
        variables: Names of the predicted quantities, one output channel each.
        feature_channels: Channel width of the decoded feature map. Wired from
            the decoder during stage construction.
        input_size: Input spatial size the values are resized to. Wired from
            the encoder. None leaves the decoder's size.
        hidden_channels: Width of an optional 3x3 conv refinement. None
            projects linearly.
        dropout: Dropout2d probability before the projection.
        units: Unit the values are in, shared by every variable. None states
            no unit.

    Raises:
        ValueError: `variables` is empty or names a variable twice.

    Examples:
        >>> head = RegressionHead(variables=["biomass"], feature_channels=256, units="t/ha")
        >>> head.variables
        ['biomass']
    """

    def __init__(
        self,
        variables: list[str],
        feature_channels: int,
        input_size: int | tuple[int, int] | None = None,
        hidden_channels: int | None = None,
        dropout: float = 0.0,
        units: str | None = None,
    ) -> None:
        if not variables:
            raise ValueError("a regression head needs at least one variable")
        repeated = sorted({name for name in variables if variables.count(name) > 1})
        if repeated:
            raise ValueError(f"variables name {repeated} more than once")
        super().__init__(
            num_classes=len(variables),
            feature_channels=feature_channels,
            input_size=input_size,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )
        self.variables = list(variables)
        self.units = units
