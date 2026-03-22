import pytest
import torch
from pathlib import Path
import torch
import lightning as L
import yaml

from geosave_engine.ai_tasks.semseg.supervised.loader import DataModule
from geosave_engine.ai_tasks.semseg.supervised.main import SemSegModel

@pytest.fixture
def mock_datamodule():
    """Sets up the DataModule once for all tests to use."""
    dataset_dir = Path("dataset/isprs_postdam").resolve()
    metadata_path = Path("dataset/isprs_postdam/metadata.yaml").resolve()
    return DataModule(data_dir=dataset_dir, metadata=metadata_path)

@pytest.fixture
def mock_model():
    """Sets up the Model once for all tests to use."""
    metadata_path = Path("dataset/isprs_postdam/metadata.yaml").resolve()
    transform_path = Path("dataset/isprs_postdam/transform.yaml").resolve()
    with metadata_path.open("r") as f:
        metadata_dict = yaml.safe_load(f)
    with transform_path.open("r") as f:
        transform_dict = yaml.safe_load(f)
    return SemSegModel(arch="dpt", optim="AdamW", loss="CELoss", 
                       metadata_dict=metadata_dict, 
                       transform_dict=transform_dict)

def test_model_forward_pass(mock_model):
    """Test ONLY the model architecture, no data loading required."""
    # Create a fake image batch (Batch=2, Channels=3, H=256, W=256)
    fake_images = torch.randn(2, 3, 256, 256)
    
    # Run a forward pass
    logits = mock_model(fake_images)
    
    # ASSERTIONS: This is how pytest knows if the logic is correct
    assert logits.shape == (2, mock_model.metadata_interpreter.nclass, 256, 256)
    assert not torch.isnan(logits).any(), "Model output contains NaNs!"

def test_training_loop(mock_model, mock_datamodule):
    """Test the Lightning Trainer loop."""
    trainer = L.Trainer(accelerator="cpu", fast_dev_run=True)
    trainer.fit(mock_model, datamodule=mock_datamodule)
    
    assert trainer.global_step > 0, "Trainer did not take any optimization steps."