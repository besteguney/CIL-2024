# Imports
import torch
import numpy as np
from parameters import *

import utils
import trainer

import dataset

import matplotlib.pyplot as plt
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset, val_dataset = dataset.load_all_data(
    ROOT_PATH,
    device,
    BATCH_SIZE,
    use_patches=False,
    val_size=VAL_SIZE,
    resize_to=None
)

class Block(torch.nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_ch),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)


class UNet(torch.nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
        self.dec_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # decoder blocks
        self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()) # 1x1 convolution for producing the output

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 1 channel


def accuracy_fn(y_pred, y_true):
    return ((y_pred > 0.5) == (y_true > 0.5)).float().mean().item()

def patches_pred(y):
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    pred = y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE)
    pred = (pred > 0.5).float().mean((-1, -3)) > CUTOFF
    return pred

def patch_accuracy_fn(y_hat: torch.Tensor, y: torch.Tensor):
    return (patches_pred(y_hat) == patches_pred(y)).float().mean().item()

def f1_score(y_pred, y_true):
    prec_array = y_true[y_pred]
    if len(prec_array) == 0:
        return 0.0
    precision = prec_array.float().mean()

    rec_array = y_pred[y_true]
    if len(rec_array) == 0:
        return 0.0
    recall = rec_array.float().mean()
    if precision == 0 or recall == 0:
        return 0.0
    return (2 * precision * recall / (precision + recall)).item()

def f1_score_fn(y_pred, y_true):
    return f1_score(y_pred > 0.5, y_true > 0.5)

def f1_score_patches_fn(y_pred, y_true):
    return f1_score(patches_pred(y_pred), patches_pred(y_true))


# def dice_loss(y_pred, y_true):
#     smooth = 1e-4
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = (y_true_f * y_pred_f).sum()
#     dice = (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
#     return 1-dice



model = UNet().to(device)
loss_fn = nn.BCELoss()
# loss_fn = dice_loss

optimizer = torch.optim.Adam(model.parameters())
metrics = {
    'acc': accuracy_fn,
    'patch_acc': patch_accuracy_fn,
    'f1_score': f1_score_fn,
    'f1_score_patches': f1_score_patches_fn
}

trainer.train(train_dataset, val_dataset, model, loss_fn, metrics, optimizer, N_EPOCHS)