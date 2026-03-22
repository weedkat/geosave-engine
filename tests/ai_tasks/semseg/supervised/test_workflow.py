import pytest
import torch
from pathlib import Path
import torch
import lightning as L
import yaml

from geosave_engine.ai_tasks.semseg.supervised.loader import DataModule
from geosave_engine.ai_tasks.semseg.supervised.main import SemSegModel

DATASET_DIR = Path("dataset/isprs_postdam").resolve()
METADATA_PATH = DATASET_DIR / "metadata.yaml"
TRANSFORM_PATH = DATASET_DIR / "transform.yaml"

with METADATA_PATH.open("r") as f:
    metadata_dict = yaml.safe_load(f)

with TRANSFORM_PATH.open("r") as f:
    transform_dict = yaml.safe_load(f)

@pytest.fixture
def mock_datamodule():
    """Sets up the DataModule once for all tests to use."""
    return DataModule(data_dir=DATASET_DIR, 
                      metadata_dict=metadata_dict, 
                      transform_dict=transform_dict,
                      batch_size=2,  # Use small batch size for testing
                      num_workers=0)  # Use 0 workers for testing to avoid multiprocessing issues

@pytest.fixture
def mock_model():
    """Sets up the Model once for all tests to use."""
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

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock

# Import your callbacks here!
from geosave_engine.ai_tasks.semseg.core.callbacks import CalibrationCallback, RGBMaskWriter, DynamicModelCheckpoint

def test_calibration_callback(mock_model):
    """Test the per-class threshold calibration logic."""
    # 1. Initialize callback and model state
    cb = CalibrationCallback(threshold_steps=10)
    mock_model.calibrating = False
    
    # Initialize a dummy buffer for thresholds
    n_classes = mock_model.metadata_interpreter.nclass
    mock_model.class_thresholds = torch.zeros(n_classes)
    
    # 2. Trigger start hook
    cb.on_test_epoch_start(trainer=None, pl_module=mock_model)
    assert mock_model.calibrating is True, "Callback failed to set calibrating flag."
    
    # 3. Create dummy batch data (Batch=2, H=64, W=64)
    # We make preds match labels perfectly so the optimal F1 threshold should be easily found
    preds = torch.randint(0, n_classes, (2, 64, 64))
    labels = preds.clone()
    max_probs = torch.rand(2, 64, 64) 
    
    outputs = (preds, max_probs, labels)
    
    # 4. Trigger batch end
    cb.on_test_batch_end(trainer=None, pl_module=mock_model, outputs=outputs, batch=None, batch_idx=0)
    assert len(cb.test_preds) == 1, "Callback did not store predictions in memory."
    
    # 5. Trigger epoch end (This runs the F1 optimization loop)
    cb.on_test_epoch_end(trainer=None, pl_module=mock_model)
    
    # 6. Assertions
    assert mock_model.calibrating is False, "Callback did not reset calibrating flag."
    assert len(cb.test_preds) == 0, "Callback did not clear memory after calibration."
    
    # Because our dummy data had perfectly matching preds and labels, 
    # the thresholds should have updated from the 0.0 default.
    assert torch.any(mock_model.class_thresholds > 0.0), "Thresholds were not updated!"


def test_rgb_mask_writer(mock_model, tmp_path):
    """Test if the writer correctly generates .pt, .png, and .tif files."""
    # tmp_path is a built-in pytest fixture that creates a temporary folder that deletes itself after the test
    cb = RGBMaskWriter(
        save_dir=str(tmp_path), 
        file_prefix="test_pred", 
        save_form=["class_pt", "rgb_png", "class_tif"]
    )
    
    # 1. Create dummy prediction data
    preds = torch.randint(0, mock_model.metadata_interpreter.nclass, (2, 64, 64))
    max_probs = torch.rand(2, 64, 64)
    ids = ["img_A", "img_B"]
    
    # Create empty dicts to simulate rasterio profiles
    meta_profiles = [{}, {}] 
    
    prediction = (preds, max_probs, ids, meta_profiles)
    
    # 2. Create a dummy Trainer object to provide the 'global_step'
    trainer_mock = MagicMock()
    trainer_mock.global_step = 42
    
    # 3. Trigger the write hook
    cb.write_on_batch_end(
        trainer=trainer_mock, 
        pl_module=mock_model, 
        prediction=prediction, 
        batch_indices=[0, 1], 
        batch=None, 
        batch_idx=7, 
        dataloader_idx=0
    )
    
    # 4. Assertions: Check if all 6 files (2 images x 3 formats) were physically written to disk!
    expected_files = [
        "test_pred_step42_batch7_idimg_A_class.pt",
        "test_pred_step42_batch7_idimg_A_rgb.png",
        "test_pred_step42_batch7_idimg_A_class.tif",
        "test_pred_step42_batch7_idimg_B_class.pt",
        "test_pred_step42_batch7_idimg_B_rgb.png",
        "test_pred_step42_batch7_idimg_B_class.tif",
    ]
    
    for file_name in expected_files:
        file_path = tmp_path / file_name
        assert file_path.exists(), f"Writer failed to save {file_name}"


def test_dynamic_checkpoint_name(mock_model):
    """Test if the custom checkpointer correctly extracts model attributes."""
    # 1. Manually set attributes on our mock model
    mock_model.arch = "resnet50"
    mock_model.dataset_name = "isprs_potsdam"
    
    # 2. Initialize callback with our custom formatting string
    cb = DynamicModelCheckpoint(filename="weights_{arch}_{dataset}_{epoch:02d}")
    
    # 3. Trigger the hook
    cb.on_fit_start(trainer=None, pl_module=mock_model)
    
    # 4. Assert that the string replacement worked perfectly
    # Note: '{epoch:02d}' should remain untouched for Lightning to fill in later!
    assert cb.filename == "weights_resnet50_isprs_potsdam_{epoch:02d}"