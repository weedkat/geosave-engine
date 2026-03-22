import torch
import lightning as L
from lightning.pytorch.callbacks import Callback

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