import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils.datasets import ImageDataset
import parameters as params
from utils import utils
import os
import train_model
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import trainer
import segmentation_models_pytorch as smp
import wandb

class SegformerForRoadSegmentation(nn.Module):
    def __init__(self, num_classes=1, input_size=(384, 384)):
        super(SegformerForRoadSegmentation, self).__init__()
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        # Ensure decode_head matches the output channels of the segformer
        '''
        self.decode_head = nn.Sequential(
            nn.Conv2d(150, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
        '''
        self.decode_head = nn.Sequential(
            nn.Conv2d(150, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

        self.input_size = input_size

    def forward(self, x):
        # Forward pass through Segformer model
        outputs = self.segformer(x)
        
        # Extract the segmentation logits directly from outputs
        seg_logits = outputs.logits
        
        # Pass through the decode head
        logits = self.decode_head(seg_logits)

        # Resize logits to match input size (if necessary)
        logits = nn.functional.interpolate(logits, size=self.input_size, mode='bilinear', align_corners=False)
        
        return logits


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images, masks = train_model.get_data(24)

    print("Finished data collection.")

    train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.1, random_state=23, shuffle=True
    )

    # reshape the image to simplify the handling of skip connections and maxpooling
    train_dataset = ImageDataset(train_images, train_masks, device, use_patches=False, resize_to=(params.RESIZE, params.RESIZE))
    val_dataset = ImageDataset(val_images, val_masks, device, use_patches=False, resize_to=(params.RESIZE, params.RESIZE))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params.BATCH_SIZE, shuffle=True)

    model = SegformerForRoadSegmentation(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)
    pos_weight = torch.tensor([1.5]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    metric_fns = {'acc': utils.accuracy_fn}

    wandb_run = wandb.init(project="cil", entity="emmy-zhou", name="segformer_2", reinit=True)
    trainer.train_smp_wandb(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, None, 100, "segformer_2", 1, wandb_run)


if __name__ == "__main__":
    main()