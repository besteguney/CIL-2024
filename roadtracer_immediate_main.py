import os
# os.environ['CUDA_ENABLE_COREDUMP_ON_EXCEPTION']='1'
os.environ['CUDA_LAUNCH_BLOCKING']='1'


# import faulthandler
# faulthandler.enable()

import argparse

# Imports
import torch
import numpy as np

import roadtracer_train
import roadtracer_dataset
from roadtracer_model1 import RoadTracerModel
from roadtracer_infer import inference_immediate
from roadtracer_logging import Logger, LogMetrics
from utils import load_all_from_path



parser = argparse.ArgumentParser()
parser.add_argument("--root_path", default="ethz-cil-road-segmentation-2024", type=str, help="Path to dataset root")

parser.add_argument("--image_size", default=256, type=int, help="The size to which we resize data before training the model")
parser.add_argument("--roadtracer_angle_samples", default=32, type=int, help="How many angles are considered for the next roadtracer step")
parser.add_argument("--roadtracer_patch_size", default=64, type=int, help="The size of the patch used as roadtracer input")
parser.add_argument("--step_distance", default=16.0, type=float, help="The length of the edges in the generated graph")
parser.add_argument("--single_angle_target", default=False, type=bool, help="Use one-hot encoding for the angle or provide a score for all of them")

parser.add_argument("--batch_size", default=8, type=int, help="The batch size used when training")
parser.add_argument("--epochs", default=10000, type=int, help="The amount of epochs to train for")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The model learning rate")

parser.add_argument("--validation_size", default=24, type=int, help="The size of the validation set (out of 144 images in the training set)")

parser.add_argument("--log_name", default="roadtracer_immediate", type=str, help="The name for logging the run (tensorboard & wandb)")
parser.add_argument("--wandb_entity", default="matezzzz", type=str, help="The name of the wandb team where the run will be logged")
parser.add_argument("--use_wandb", default=True, type=bool, help="Whether to use wandb (weights and biases) or not")

parser.add_argument("--load_model", default="log/roadtracer_immediate-24-07-2024_19-40-26/model_1899.pt", type=str, help="The model to use for inference")



def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = RoadTracerModel(args.roadtracer_patch_size, input_channels=3, num_angles=args.roadtracer_angle_samples).to(device)
    
    if args.load_model is None:
        train_images, val_images = roadtracer_dataset.load_all_data(
            args.root_path,
            args.validation_size,
            args.image_size
        )

        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        logger = Logger(args.log_name, args.use_wandb, args.wandb_entity, vars(args))
        train_loop = roadtracer_train.ImmediateTrainingLoop(
            logger, train_images, val_images,
            args.batch_size, args.epochs, args.roadtracer_angle_samples, args.roadtracer_patch_size, args.image_size, 
            args.step_distance, args.single_angle_target
        )
        train_loop.train(model, optimizer)
    else:
        model.load_state_dict(torch.load(args.load_model))
        inference_immediate(model, load_all_from_path(f"{args.root_path}/test/images/"),
            args.image_size, args.roadtracer_patch_size, args.roadtracer_angle_samples
        )






if __name__ == "__main__":
    main(parser.parse_args())
  
# TODO !
# Use the additional loss term
# Smaller batch size? orthogonal roads seem to affect the model too much when there are several batches of those
# Terminate if action = move and probs are low