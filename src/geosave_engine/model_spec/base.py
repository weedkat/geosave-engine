"""Shared rules for portable model specifications."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Text = Annotated[str, StringConstraints(min_length=1)]


class SpecModel(BaseModel):
    """Reject unknown settings and non-finite numerical parameters."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
    )
