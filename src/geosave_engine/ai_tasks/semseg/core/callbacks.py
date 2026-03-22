import torch
import numpy as np
import lightning as L
import rasterio
from PIL import Image
from lightning.pytorch.callbacks import (Callback, 
                                         BasePredictionWriter, 
                                         ModelCheckpoint, 
                                         EarlyStopping,
                                         LearningRateMonitor,
                                         DeviceStatsMonitor,
                                         RichProgressBar)

from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from pathlib import Path


class CalibrationCallback(Callback):
    """
    Calibrates per-class confidence thresholds to maximize metrics on test set.
    Updates the pl_module.class_thresholds buffer.
    """
    def __init__(self, 
                 threshold_begin: float = 0.0, 
                 threshold_end: float = 1.0, 
                 threshold_steps: int = 100, 
                 metric_name: str = "f1", # F1 is often better for per-class thresholds
                 ):
        assert 0.0 <= threshold_begin < threshold_end <= 1.0, "Threshold range must be within [0.0, 1.0]"

        super().__init__()
        self.threshold_range = torch.linspace(threshold_begin, threshold_end, threshold_steps)
        self.metric_name = metric_name
        
        # Lists to collect CPU tensors
        self.test_preds = []
        self.test_max_probs = []
        self.test_labels = []
    
    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.calibrating = True
        # Clear lists just in case
        self.test_preds.clear()
        self.test_max_probs.clear()
        self.test_labels.clear()
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        preds, max_probs, labels = outputs  
        
        self.test_preds.append(preds.detach().cpu())
        self.test_max_probs.append(max_probs.detach().cpu())
        self.test_labels.append(labels.detach().cpu())
    
    def on_test_epoch_end(self, trainer, pl_module):
        if not self.test_preds:
            pl_module.calibrating = False
            return
        
        all_preds = torch.cat(self.test_preds).view(-1)
        all_max_probs = torch.cat(self.test_max_probs).view(-1)
        all_labels = torch.cat(self.test_labels).view(-1)
        
        n_classes = pl_module.metadata_interpreter.nclass
        new_thresholds = torch.zeros(n_classes, device=pl_module.device)

        print(f"\n--- Starting Per-Class Calibration (Optimizing {self.metric_name}) ---")

        # 2. Iterate through each class to find its specific best threshold
        for c in range(n_classes):
            # Isolate pixels where the model predicted this class
            class_mask = (all_preds == c)
            
            if not class_mask.any():
                new_thresholds[c] = 0.5 # Default fallback
                continue
                
            c_probs = all_max_probs[class_mask]
            c_labels = all_labels[class_mask]
            
            # Binary ground truth: was it actually the predicted class?
            is_correct = (c_labels == c)
            
            best_t = 0.0
            best_score = -1.0

            # 3. Inner search loop for this class
            for t in self.threshold_range:
                # Basic F1 calculation (vectorized for speed)
                stays_correct = (c_probs >= t) & is_correct
                stays_wrong = (c_probs >= t) & (~is_correct)
                rejected_correct = (c_probs < t) & is_correct
                
                tp = stays_correct.sum().float()
                fp = stays_wrong.sum().float()
                fn = rejected_correct.sum().float()
                
                precision = tp / (tp + fp + 1e-7)
                recall = tp / (tp + fn + 1e-7)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
                
                if f1 > best_score:
                    best_score = f1
                    best_t = t.item()

            new_thresholds[c] = best_t
            print(f"Class {c} ({pl_module.metadata_interpreter.class_names[c]}): T={best_t:.3f}, F1={best_score:.3f}")

        # 4. Update the model's registered buffer
        pl_module.class_thresholds.copy_(new_thresholds)
        
        # Cleanup
        self.test_preds.clear()
        self.test_max_probs.clear()
        self.test_labels.clear()
        pl_module.calibrating = False


class RGBMaskWriter(BasePredictionWriter):
    """
    Writes predicted segmentation masks to disk after each validation/test batch.
     - Expects outputs to be (preds, max_probs, ids) where:
        - preds: (B, H, W) predicted class indices
        - max_probs: (B, H, W) confidence of the predicted class
        - ids: (B,) identifiers or filenames for the images
    """
    def __init__(self, save_dir: str = "predictions", file_prefix: str = "pred", save_form: list = ["class", "rgb", "tiff"]):
        # BasePredictionWriter STRICTLY requires the write_interval argument
        super().__init__(write_interval="batch") 
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.file_prefix = file_prefix
        self.save_form = save_form

    # The correct method name for BasePredictionWriter
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        # 1. Unpack what your predict_step returned
        preds, max_probs, ids = prediction
        
        # 2. Prepare the data (Move to CPU once)
        preds_cpu = preds.cpu()
        class_pred_np = preds_cpu.numpy().astype(np.uint8)

        # Generate RGB only if the user asked for PNGs OR TIFFs
        if "rgb" in self.save_form or "tiff" in self.save_form:
            rgb_pred_np = pl_module.metadata_interpreter.class_to_rgb(class_pred_np) # (B, H, W, 3)

        # 3. Save individual images in the batch
        for i in range(preds.shape[0]):
            # Create a clean base filename template
            img_id = ids[i] if ids is not None else f"idx_{batch_indices[i]}"
            base_name = self.save_dir / f"{self.file_prefix}_step{trainer.global_step}_batch{batch_idx}_id{img_id}"

            # Option A: Save raw class indices as a PyTorch Tensor
            if "class" in self.save_form:
                # Saving the tensor directly is cleaner than wrapping a numpy array
                torch.save(preds_cpu[i], f"{base_name}_class.pt")
            
            # Option B: Save as a standard RGB PNG
            if "rgb" in self.save_form:
                # PIL easily converts numpy arrays to standard image files
                img_pil = Image.fromarray(rgb_pred_np[i])
                img_pil.save(f"{base_name}_rgb.png")
            
            # Option C: Save as a GeoTIFF
            if "tiff" in self.save_form:
                # Rasterio strictly expects (Channels, Height, Width)
                tiff_data = rgb_pred_np[i].transpose(2, 0, 1)
                
                with rasterio.open(
                    f"{base_name}_rgb.tif", 
                    'w', 
                    driver='GTiff', 
                    height=tiff_data.shape[1], 
                    width=tiff_data.shape[2], 
                    count=3, 
                    dtype=tiff_data.dtype
                ) as dst:
                    dst.write(tiff_data)

class DynamicModelCheckpoint(ModelCheckpoint):
    # if no __init__ python assume to super().__init__()
    def on_fit_start(self, trainer, pl_module):
        arch = getattr(pl_module, "arch", "unknown_arch")
        dataset = getattr(pl_module, "dataset_name", "unknown_data")
        
        if self.filename:
            self.filename = self.filename.replace("{arch}", str(arch))
            self.filename = self.filename.replace("{dataset}", str(dataset))
            
        # call the parent class so standard Lightning logic runs!
        super().on_fit_start(trainer, pl_module)


early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=5,
    verbose=True,
    mode="min"
)

# Create a beautiful, custom-colored progress bar
beautiful_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="bold cyan",           # The text (e.g., "Epoch 1")
        progress_bar="green",              # The bar filling up
        progress_bar_finished="bold blue", # Color when epoch is done
        progress_bar_pulse="bold purple",  # Color during indeterminate steps
        batch_progress="yellow",           # The "100/1000" batch counter
        time="dim white",                  # Time remaining
        processing_speed="dim white",      # it/s
        metrics="bold cyan"                # The actual loss/accuracy numbers!
    )
)