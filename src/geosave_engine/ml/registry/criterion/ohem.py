import torch
import torch.nn.functional as F
from torch import nn


# Adapted from https://github.com/charlesCXK/TorchSemiSeg/blob/main/furnace/seg_opr/loss_opr.py
class ProbOhemCrossEntropy2d(nn.Module):
    """Cross-entropy over the hardest pixels only (online hard example mining).

    Pixels whose predicted probability for their own label exceeds `thresh`
    are dropped from the loss, so training concentrates on the pixels the
    model still gets wrong. At least `min_kept` pixels always survive.

    Args:
        ignore_index: Label excluded from the loss.
        reduction: Reduction passed to `torch.nn.CrossEntropyLoss`.
        thresh: Probability above which a correctly classified pixel is dropped.
        min_kept: Lower bound on the pixels entering the loss.
        weight: Per-class rescaling weights passed to `torch.nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        ignore_index: int,
        reduction: str = "mean",
        thresh: float = 0.7,
        min_kept: int = 256,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = nn.CrossEntropyLoss(
            reduction=reduction, weight=weight, ignore_index=ignore_index
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Score predictions against labels over the mined pixels.

        Args:
            pred: (B, C, H, W) class logits.
            target: (B, H, W) class labels.

        Returns:
            Scalar loss over the kept pixels.
        """
        b, c, h, w = pred.size()
        target = target.view(-1)  # (B*H*W,)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)  # (B, C, H, W)
        prob = (prob.transpose(0, 1)).reshape(c, -1)  # (C, B*H*W)

        # Below min_kept valid pixels there is nothing to mine.
        if 0 < num_valid and self.min_kept <= num_valid:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)  # (B, H, W)

        return self.criterion(pred, target)
