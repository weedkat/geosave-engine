"""Shared validation policy for deployable workflow inputs."""

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Reject unknown fields and non-finite deployment parameters."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
    )
