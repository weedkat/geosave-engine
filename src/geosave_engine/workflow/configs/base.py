"""Shared validation rules for primitive workflow parameters."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]


class ConfigModel(BaseModel):
    """Reject unknown fields and non-finite deployment parameters."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
    )
