import torch
from lightning.pytorch.callbacks import Callback 

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
