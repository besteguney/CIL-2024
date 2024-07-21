import os
os.environ['CUDA_ENABLE_COREDUMP_ON_EXCEPTION']='1'
import argparse

# Imports
import torch
import numpy as np
from parameters import *

import roadtracer_train
import roadtracer_dataset
from roadtracer_model1 import RoadTracerModel
from roadtracer_logging import Logger, LogMetrics

# import faulthandler
# faulthandler.enable()


parser = argparse.ArgumentParser()
parser.add_argument("--root_path", default="ethz-cil-road-segmentation-2024", type=str, help="Path to dataset root")

parser.add_argument("--roadtracer_angle_samples", default=64, type=int, help="How many angles are considered for the next roadtracer step")
parser.add_argument("--roadtracer_patch_size", default=128, type=int, help="The size of the patch used as roadtracer input")
parser.add_argument("--step_distance", default=10.0, type=float, help="The length of the edges in the generated graph")
parser.add_argument("--merge_distance", default=10.0/1.3, type=float, help="The closest two points in the generated graph can be without getting merged")
parser.add_argument("--single_angle_target", default=False, type=bool, help="Use one-hot encoding for the angle or provide a score for all of them")
parser.add_argument("--max_graph_size", default=1000, type=int, help="The largest amount of vertices allowed in a graph")

parser.add_argument("--batch_size", default=8, type=int, help="The batch size used when training")
parser.add_argument("--epochs", default=10, type=int, help="The amount of epochs to train for")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The model learning rate")

parser.add_argument("--validation_size", default=24, type=int, help="The size of the validation set (out of 144 images in the training set)")

parser.add_argument("--log_name", default="roadtracer_graph", type=str, help="The name for logging the run (tensorboard & wandb)")
parser.add_argument("--wandb_entity", default="abjarnsteins-eth-z-rich", type=str, help="The name of the wandb team where the run will be logged")
parser.add_argument("--use_wandb", default=True, type=bool, help="Whether to use wandb (weights and biases) or not")



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_images, val_images = roadtracer_dataset.load_all_data(
        args.root_path,
        args.validation_size
    )
    
    logger =  Logger(args.log_name, args.use_wandb, args.wandb_entity, vars(args))
    model = RoadTracerModel(args.roadtracer_patch_size, num_angles=args.roadtracer_angle_samples).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    train_loop = roadtracer_train.GraphTrainingLoop(
        logger, train_images, val_images,
        args.batch_size, args.epochs, args.roadtracer_angle_samples, args.roadtracer_patch_size,
        args.step_distance, args.merge_distance, args.single_angle_target, args.max_graph_size
    )
    train_loop.train(model, optimizer)



if __name__ == "__main__":
    main(parser.parse_args())
  
# TODO !
# Use the additional loss term
# Smaller batch size? orthogonal roads seem to affect the model too much when there are several batches of those
# Terminate if action = move and probs are low