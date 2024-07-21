# Imports
import torch
import numpy as np
from parameters import *

import roadtracer_train
import roadtracer_dataset
from roadtracer_model1 import RoadTracerModel

# import faulthandler
# faulthandler.enable()

roadtracer_angle_samples = 64
roadtracer_patch_size = 128
batch_size = 8
epochs = 10
step_distance = 10.0
merge_distance = step_distance / 1.3


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_images, val_images = roadtracer_dataset.load_all_data(
    ROOT_PATH,
    VAL_SIZE
)

model = RoadTracerModel(roadtracer_patch_size, num_angles=roadtracer_angle_samples).to(device)
optimizer = torch.optim.Adam(model.parameters())

train_loop = roadtracer_train.GraphTrainingLoop(
    train_images, val_images,
    batch_size, epochs, roadtracer_angle_samples, roadtracer_patch_size,
    step_distance, merge_distance
)
train_loop.train(model, optimizer)

  

# TODO !
# Use the additional loss term
# Train on all scores (target is not one-hot)
# Smaller batch size? orthogonal roads seem to affect the model too much when there are several batches of those
