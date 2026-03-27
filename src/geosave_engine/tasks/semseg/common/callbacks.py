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
    Calibrates per-class confidence thresholds to maximize metrics on the VALIDATION set.
    Updates the pl_module.class_thresholds buffer.
    """
    def __init__(self, threshold_begin: float = 0.0, threshold_end: float = 1.0, threshold_steps: int = 100, metric_name: str = "f1"):
        super().__init__()
        self.threshold_range = torch.linspace(threshold_begin, threshold_end, threshold_steps)
        self.metric_name = metric_name
        
        self.val_preds = []
        self.val_max_probs = []
        self.val_labels = []
    
    def on_validation_epoch_start(self, trainer, pl_module):
        # Only calibrate during the sanity check or actual validation epochs
        pl_module.calibrating = True
        self.val_preds.clear()
        self.val_max_probs.clear()
        self.val_labels.clear()
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # NOTE: Ensure your pl_module.validation_step returns these 3 items!
        preds, max_probs, labels = outputs  
        
        self.val_preds.append(preds.detach().cpu())
        self.val_max_probs.append(max_probs.detach().cpu())
        self.val_labels.append(labels.detach().cpu())
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.val_preds:
            pl_module.calibrating = False
            return
        
        all_preds = torch.cat(self.val_preds).view(-1)
        all_max_probs = torch.cat(self.val_max_probs).view(-1)
        all_labels = torch.cat(self.val_labels).view(-1)
        
        n_classes = pl_module.metadata_interpreter.nclass
        new_thresholds = torch.zeros(n_classes, device=pl_module.device)

        # 1. Inner search loop, we optimize f1 score for each class independently
        for c in range(n_classes):
            class_mask = (all_preds == c)
            if not class_mask.any():
                new_thresholds[c] = 0.5 
                continue
                
            c_probs = all_max_probs[class_mask]
            c_labels = all_labels[class_mask]
            is_correct = (c_labels == c)
            
            best_t, best_score = 0.0, -1.0

            for t in self.threshold_range:
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

        # 2. Update the model's thresholds BEFORE the next training epoch starts
        pl_module.class_thresholds.copy_(new_thresholds)
        
        self.val_preds.clear()
        self.val_max_probs.clear()
        self.val_labels.clear()
        pl_module.calibrating = False


class MaskWriter(BasePredictionWriter):
    def __init__(self, save_dir: str = "predictions", file_prefix: str = "pred", save_form: list = ["class_tif"]):
        super().__init__(write_interval="batch") 
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.file_prefix = file_prefix
        self.save_form = save_form

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        # 1. Unpack the predictions AND the spatial metadata
        # We assume meta_profiles is a list of rasterio profile dictionaries
        preds, max_probs, ids, meta_profiles = prediction
        
        preds_cpu = preds.cpu()
        class_pred_np = preds_cpu.numpy().astype(np.uint8)

        # Generate RGB only if the user specifically asked for PNGs
        if "rgb_png" in self.save_form:
            rgb_pred_np = pl_module.metadata_interpreter.class_to_rgb(class_pred_np)

        for i in range(preds.shape[0]):
            img_id = ids[i] if ids is not None else f"idx_{batch_indices[i]}"
            base_name = self.save_dir / f"{self.file_prefix}_step{trainer.global_step}_batch{batch_idx}_id{img_id}"

            # Option A: Save raw tensor
            if "class_pt" in self.save_form:
                torch.save(preds_cpu[i], f"{base_name}_class.pt")
            
            # Option B: Save visual RGB as PNG (for quick human inspection)
            if "rgb_png" in self.save_form:
                img_pil = Image.fromarray(rgb_pred_np[i])
                img_pil.save(f"{base_name}_rgb.png")
            
            # Option C: Save analytical Class Index as GeoTIFF (for GIS software)
            if "class_tif" in self.save_form:
                # 1. Grab the original spatial profile for this specific image
                out_meta = meta_profiles[i].copy() if meta_profiles else {}
                
                # 2. Overwrite the dimensions/types to match our prediction
                # We do this because the input might have had 4 bands, but our mask has 1
                out_meta.update({
                    "driver": "GTiff",
                    "height": class_pred_np.shape[1],
                    "width": class_pred_np.shape[2],
                    "count": 1,                   # STRICTLY 1 channel for class index
                    "dtype": rasterio.uint8,      # Save space with uint8
                    "nodata": pl_module.metadata_interpreter.ignore_index # Tell GIS to ignore this value
                })
                
                # 3. Write the single-channel class prediction
                with rasterio.open(f"{base_name}_class.tif", 'w', **out_meta) as dst:
                    # Rasterio expects a (Channels, Height, Width) shape.
                    # np.newaxis converts (H, W) to (1, H, W)
                    dst.write(class_pred_np[i][np.newaxis, ...])
                    

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