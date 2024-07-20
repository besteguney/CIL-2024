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



model = RoadTracerModel(roadtracer_distance_samples).to(device)
# loss_fn = nn.BCELoss()
# loss_fn = dice_loss

optimizer = torch.optim.Adam(model.parameters())
metrics = {
    # 'acc': accuracy_fn,
    # 'patch_acc': patch_accuracy_fn,
    # 'f1_score': f1_score_fn,
    # 'f1_score_patches': f1_score_patches_fn
}

roadtracer_train.train(train_dataset, val_dataset, model, loss_fn, metrics, optimizer, N_EPOCHS)