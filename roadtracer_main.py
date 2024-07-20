# Imports
import torch
import numpy as np
from parameters import *

import utils
import roadtracer_train

import roadtracer_dataset

import matplotlib.pyplot as plt
from torch import nn
from roadtracer_model import RoadTracerModel



roadtracer_angle_samples = 256
roadtracer_view_radius = 60
roadtracer_sample_radius = 10
roadtracer_distance_samples = 64
batch_size = 8


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset, val_dataset = roadtracer_dataset.load_all_data(
    ROOT_PATH,
    device,
    batch_size,
    VAL_SIZE,
    roadtracer_angle_samples,
    roadtracer_distance_samples,
    roadtracer_sample_radius,
    roadtracer_view_radius
)



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



model = RoadTracerModel(roadtracer_distance_samples).to(device)
loss_fn = nn.BCELoss()
# loss_fn = dice_loss

optimizer = torch.optim.Adam(model.parameters())
metrics = {
    # 'acc': accuracy_fn,
    # 'patch_acc': patch_accuracy_fn,
    # 'f1_score': f1_score_fn,
    # 'f1_score_patches': f1_score_patches_fn
}

roadtracer_train.train(train_dataset, val_dataset, model, loss_fn, metrics, optimizer, N_EPOCHS)