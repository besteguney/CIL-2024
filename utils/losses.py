import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight=None):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        bce_loss = self.bce_loss(logits, targets)
        combined_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return combined_loss