"""
CLI tool for calibrating confidence threshold to maximize mIoU.
"""

import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from geomoka.model.segmentation.base import SegmentationModel
from geomoka.dataloader.build import build_segmentation_dataset
from geomoka.eval.metrics import compute_iou


def calibrate_confidence_threshold(
    model_wrapper: SegmentationModel,
    dataloader: DataLoader,
    mode: str = 'sliding_window',
    threshold_range: tuple = (0.0, 0.95),
    num_steps: int = 20,
    save_plot: str = 'tests/output/confidence_calibration.png',
    reject_class: int = 255
) -> dict:
    """
    Calibrate confidence threshold by finding the value that maximizes mIoU on validation data.
    Efficient: runs inference once, then tests multiple thresholds on cached predictions.
    
    Note: Does NOT use ignore_index - calibration evaluates ALL pixels including background
    to find the threshold that best handles uncertain/low-confidence regions.
    
    Args:
        model_wrapper: SegmentationModel instance with loaded model
        dataloader: DataLoader with (images, labels) pairs
        mode: 'resize' or 'sliding_window'
        threshold_range: (min, max) threshold values to test
        num_steps: Number of threshold values to test
        save_plot: Path to save the calibration curve
        reject_class: Class index for rejected predictions
        
    Returns:
        Dictionary with best_threshold, best_miou, and plot path
    """
    inference = model_wrapper.inference
    num_classes = inference.num_classes
    
    print(f'[Calibration] Running inference on validation set...')
    
    # Store original threshold and disable rejection during inference
    original_threshold = inference.confidence_threshold
    inference.confidence_threshold = 0.0  # No rejection during initial inference
    
    # Run inference ONCE and cache results
    all_raw_preds = []
    all_max_confs = []
    all_targets = []
    
    for images, labels in dataloader:
        pred, conf, _ = inference(images, mode=mode, verbose=False)
        
        # Extract raw predictions (before rejection) and max confidence
        max_conf = np.max(conf, axis=-1)  # (B, H, W)
        
        # Handle batch dimension
        if pred.ndim == 3:  # (B, H, W)
            for i in range(pred.shape[0]):
                all_raw_preds.append(pred[i])
                all_max_confs.append(max_conf[i])
                all_targets.append(labels[i].cpu().numpy())
        else:  # (H, W)
            all_raw_preds.append(pred)
            all_max_confs.append(max_conf)
            all_targets.append(labels.cpu().numpy())
    
    # Stack all predictions
    all_raw_preds = np.stack(all_raw_preds)
    all_max_confs = np.stack(all_max_confs)
    all_targets = np.stack(all_targets)
    
    print(f'[Calibration] Testing {num_steps} thresholds from {threshold_range[0]:.2f} to {threshold_range[1]:.2f}')
    
    # Generate threshold values to test
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)
    mious = []
    
    # Test each threshold by applying rejection to cached predictions
    for threshold in thresholds:
        # Apply confidence rejection to cached predictions
        preds_rejected = all_raw_preds.copy()
        preds_rejected[all_max_confs < threshold] = reject_class
        
        # Compute IoU per class (no ignore_index - evaluate everything)
        iou_per_class = compute_iou(
            torch.from_numpy(preds_rejected),
            torch.from_numpy(all_targets),
            num_classes=num_classes,
            ignore_index=None
        )
        miou = float(iou_per_class.mean())
        mious.append(miou)
        
        print(f'[Calibration] Threshold={threshold:.3f} -> mIoU={miou:.4f}')
    
    # Find best threshold
    best_idx = np.argmax(mious)
    best_threshold = float(thresholds[best_idx])
    best_miou = float(mious[best_idx])
    
    # Restore original threshold
    inference.confidence_threshold = original_threshold
    
    # Plot calibration curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds, mious, 'b-', linewidth=2, label='mIoU')
    ax.axvline(best_threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Best threshold: {best_threshold:.3f} (mIoU={best_miou:.4f})')
    ax.scatter([best_threshold], [best_miou], color='red', s=100, zorder=5)
    
    ax.set_xlabel('Confidence Threshold', fontsize=12)
    ax.set_ylabel('mIoU', fontsize=12)
    ax.set_title('Confidence Threshold Calibration Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    save_path = Path(save_plot)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f'\n[Calibration] ✓ Best threshold: {best_threshold:.3f} with mIoU: {best_miou:.4f}')
    print(f'[Calibration] ✓ Calibration curve saved to {save_path}')
    print(f'\nTo use this threshold in inference:')
    print(f'  inference.confidence_threshold = {best_threshold:.3f}')
    
    return {
        'best_threshold': best_threshold,
        'best_miou': best_miou,
        'all_thresholds': thresholds.tolist(),
        'all_mious': mious,
        'plot_path': str(save_path)
    }


def main():
    parser = argparse.ArgumentParser(description='Calibrate confidence threshold for semantic segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to training config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='sliding_window', choices=['resize', 'sliding_window'],
                        help='Inference mode (default: sliding_window)')
    parser.add_argument('--threshold-range', type=float, nargs=2, default=[0.0, 0.95],
                        help='Min and max threshold values to test (default: 0.0 0.95)')
    parser.add_argument('--num-steps', type=int, default=20,
                        help='Number of threshold values to test (default: 20)')
    parser.add_argument('--output', type=str, default='tests/output/confidence_calibration.png',
                        help='Path to save calibration curve plot')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for validation DataLoader (default: 4)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print(f'[Calibration] Loading model from {args.checkpoint}...')
    model = SegmentationModel.load(args.checkpoint)
    
    # Build validation dataset
    print(f'[Calibration] Building validation dataset...')
    dataset_cfg = config.get('dataset', {})
    val_dataset = build_segmentation_dataset(
        dataset_type=dataset_cfg.get('type', 'SegmentationDataset'),
        data_path=dataset_cfg.get('data_path'),
        split_csv=dataset_cfg.get('split', {}).get('val'),
        color_yaml=dataset_cfg.get('color_yaml'),
        transform_cfg=config.get('transforms', {}).get('val'),
        include_mask=True
    )
    
    # Create DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run calibration
    result = calibrate_confidence_threshold(
        model_wrapper=model,
        dataloader=val_loader,
        mode=args.mode,
        threshold_range=tuple(args.threshold_range),
        num_steps=args.num_steps,
        save_plot=args.output,
        reject_class=255
    )
    
    # Save results to JSON
    import json
    output_json = Path(args.output).with_suffix('.json')
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[Calibration] ✓ Results saved to {output_json}')


if __name__ == '__main__':
    main()
