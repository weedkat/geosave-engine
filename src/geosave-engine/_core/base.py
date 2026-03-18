from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

@runtime_checkable
class BaseMetadataInterpreter(Protocol):
    """Interface for interpreting dataset metadata."""
    def __init__(self, metadata: Dict[str, Any]):
        ...

@runtime_checkable
class BaseModel(Protocol):
    """Task-agnostic model interface."""

    model: Any # The actual model instance (e.g. PyTorch nn.Module)
    model_id: str # Unique identifier for the model architecture
    model_spec: Dict[str, Any] # Model specification to reconstruct architecture (e.g. model name, encoder, weights, etc.)
    metadata: Dict[str, Any]# Dataset metadata for informed training/inference

    def __init__(self, *args, **kwargs):
        """Initialize the model with architecture and metadata.
        General pattern:
        1. validate input
        2. store parameters
        3. generate model_id and model_spec
        4. build instance (e.g. PyTorch model)
        """
        ...

    def compile(self, *args, force=False, **kwargs) -> None:
        """Prepare trainer and inferencer based on the model and metadata.
        reset any existing trainer/inferencer if force=True.
        General pattern:
        1. validate input and check if already compiled (unless force=True)
        2. store compile-time configuration (e.g. transform_cfg, compile_cfg)
        3. build trainer using registry and store as self.trainer
        4. build inferencer and store as self.inferencer
        """
        ...

    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """Train the model. Return training history.
        General pattern:
        1. check that model is compiled (e.g. self.trainer is not None)
        2. try calling self.trainer.train() and return history
        3. catch and raise any exceptions with informative messages (e.g. missing parameters, incompatible trainer, etc.)
        """
        ...
    
    def predict(self, *args, **kwargs) -> Any:
        """Run inference
        General pattern:
        1. check that inferencer is available (e.g. self.inferencer is not None)
        2. delegate to self.inferencer.infer() and return predictions
        """
        ...

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """Evaluate the model and return metrics.
        General pattern:
        1. check that inferencer is available (e.g. self.inferencer is not None)
        2. delegate to an evaluator (could be self.inferencer or a separate self.evaluator) and return metrics
        """
        ...

    @classmethod
    def load(cls, load_path: str | Path) -> "BaseModel":
        """Load model artifact.
        1. load state
        2. build model instance using loaded state
        3. modify model instance (e.g. load state dict, compile)
        4. return model instance
        """
        ...

    def save(self, save_path: str | Path) -> None:
        """Save model artifact.
        1. gather necessary state (e.g. model state dict, metadata, model_spec, etc.)
        2. save state to disk (e.g. torch.save for PyTorch models)
        """
        ...
    
    def summary(self) -> None:
        """Return a summary of the model architecture and parameters."""
        print("Model Summary:")
        print(f"Model ID: {self.model_id}")
        print(f"Model Spec: {self.model_spec}")
        if self.metadata:
            print(f"Metadata: {self.metadata}")


@runtime_checkable
class BaseTrainer(Protocol):
    """Lifecycle orchestrator interface for training loops."""

    model: BaseModel

    def __init__(self, model: BaseModel, *args, **kwargs):
        """Initialize the trainer with the model and any relevant configuration.
        1. validate parameters
        2. store model and configuration parameters
        3. store parameters
        """
        ...

    def train(self, *args, **kwargs) -> dict:
        """Run training lifecycle. returns training history."""
        ...


class BaseInferencer(Protocol):
    """Interface for model inference methods."""
    model: BaseModel

    def infer(self, *args, **kwargs) -> Any:
        """Run inference and return raw predictions and confidence scores."""
        ...


class BaseEvaluator(Protocol):
    """Interface for model evaluation methods."""
    model: BaseModel

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """Evaluate the model and return metrics."""
        ...
    