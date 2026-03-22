# src/geosave_engine/ai_tasks/semseg/supervised/callbacks.py
import torch
from lightning.pytorch.callbacks import Callback


class CalibrationCallback(Callback):
    """
    Calibrates confidence threshold to maximize mIoU on test set.
    Runs after test epochs and finds the optimal threshold.
    """
    def __init__(self, 
                 threshold_begin: float = 0.0, 
                 threshold_end: float = 1.0, 
                 threshold_steps: int = 100, 
                 metric_name: str = "iou"):
        """
        Args:
            threshold_begin: Start of threshold search range (default 0.0)
            threshold_end: End of threshold search range (default 1.0)
            threshold_steps: Number of steps to search (default 100)
            metric_name: Name of the metric to optimize (default "iou")
        """
        super().__init__()
        self.threshold_begin = threshold_begin
        self.threshold_end = threshold_end
        self.threshold_steps = threshold_steps
        self.metric_name = metric_name
        self.test_preds = []
        self.test_max_probs = []
        self.test_labels = []
    
    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.calibrating = True
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect predictions during test"""
        preds, max_probs, labels = outputs  # Unpack logits and labels from test_step output
        
        self.test_preds.append(preds.detach().cpu())
        self.test_max_probs.append(max_probs.detach().cpu())
        self.test_labels.append(labels.detach().cpu())
    
    def on_test_epoch_end(self, trainer, pl_module):
        """Find best threshold after test epoch"""
        if not self.test_preds:
            print("Warning: No test predictions collected. Calibration skipped.")
            pl_module.calibrating = False
            return
        
        # Stack all batches
        all_preds = torch.stack(self.test_preds)
        all_max_probs = torch.stack(self.test_max_probs)
        all_labels = torch.stack(self.test_labels)
        
        # Find best threshold
        best_threshold, best_iou = self._find_best_threshold(all_max_probs, all_labels, pl_module)
        
        # Update model threshold
        pl_module.threshold = torch.tensor(best_threshold, device=pl_module.device)
        
        # Log results
        pl_module.log("calibrated_threshold", best_threshold)
        pl_module.log("best_iou_at_threshold", best_iou)
        
        print(f"Calibration: threshold={best_threshold:.4f}, mIoU={best_iou:.4f}")
        
        # Clear for next calibration
        self.test_preds.clear()
        self.test_max_probs.clear()
        self.test_labels.clear()

        pl_module.calibrating = False
    
    def _find_best_threshold(self, probs, labels, pl_module):
        """Search for threshold that maximizes mIoU"""
        best_threshold = 0.0
        best_metric = -float('inf')
        
        for threshold in torch.linspace(self.threshold_begin, self.threshold_end, steps=self.threshold_steps):
            preds = (probs > threshold).long().argmax(dim=1)  # (N, H, W)
            
            # Compute metrics
            pl_module.test_metrics.update(preds, labels)
            metrics = pl_module.test_metrics.compute()
            metric = metrics.get(f'test/{self.metric_name}', 0.0)
            
            if metric > best_metric:
                best_metric = metric
                best_threshold = threshold.item()
            
            pl_module.test_metrics.reset()
        
        return best_threshold, best_metric
    
