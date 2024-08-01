import argparse
import glob
import os

import cv2
from math import sin, cos, pi
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
import numpy as np
from roadtracer_utils import get_patch
from PIL import Image
import torch
from tqdm import tqdm

from roadtracer_utils import to_pytorch_img, get_circle_sample_points
from roadtracer_model1 import RoadTracerModel


parser = argparse.ArgumentParser()
parser.add_argument("--file_path", default="ethz-cil-road-segmentation-2024/test/images", type=str, help="Path to images to run inference for")
parser.add_argument("--results_path", default="roadtracer_results_128_test", type=str, help="Path to save results at")


parser.add_argument("--image_size", default=256, type=int, help="The size to which we resize data before training the model")
parser.add_argument("--angle_samples", default=32, type=int, help="How many angles are considered for the next roadtracer step")
parser.add_argument("--patch_size", default=64, type=int, help="The size of the patch used as roadtracer input")
parser.add_argument("--step_distance", default=16.0, type=float, help="The length of the edges in the generated graph")

parser.add_argument("--batch_size", default=64, type=int, help="The batch size used when training")
parser.add_argument("--rotation_samples", default=1, type=int, help="The batch size used when training")
parser.add_argument("--position_samples", default=128, type=int, help="How many points should be tested in one dimension")


parser.add_argument("--validation_size", default=24, type=int, help="The size of the validation set (out of 144 images in the training set)")

parser.add_argument("--load_model", default="models/model_1899.pt", type=str, help="The model to use for inference")




def plot_inferred(action_image, angle_image, coords, image):
    actions_taken = action_image > 0.5

    fig, ax = plt.subplots()
    ax.scatter(coords[actions_taken][..., 0], coords[actions_taken][..., 1], color="red")
    ax.scatter(coords[np.logical_not(actions_taken)][..., 0], coords[np.logical_not(actions_taken)][..., 1], color="green")
    
    lines = []
    for action_row, angle_row, coord_row in zip(action_image, angle_image, coords):
        for action, angles, coords in zip(action_row, angle_row, coord_row):
            if action > 0.5:
                best_pts = get_circle_sample_points(coords, args.angle_samples, args.step_distance)[np.argsort(angles)[::-1]][:5]
                for p in best_pts:
                    lines.append((coords, p))
    lc = LineCollection(lines)
    ax.add_collection(lc)

    ax.imshow(image)
    plt.show()



def inference_immediate(image_full_size, model, args):
    image = cv2.resize(image_full_size, (args.image_size, args.image_size))

    sample_coords_1d = np.linspace(0, args.image_size, args.position_samples)
    coords = np.stack(np.meshgrid(sample_coords_1d, sample_coords_1d), 2)
    coords_batched = np.reshape(coords, [-1, args.batch_size, 2])

    sample_indices_1d = np.arange(args.position_samples)
    sample_idx = np.stack(np.meshgrid(sample_indices_1d, sample_indices_1d), 2)
    sample_idx_batched = np.reshape(sample_idx, [-1, args.batch_size, 2])

    with tqdm(ncols=150, total=args.rotation_samples * len(coords_batched)) as pbar:
        action_image = np.zeros([args.position_samples, args.position_samples])
        angle_image = np.zeros([args.position_samples, args.position_samples, args.angle_samples])
        for rot_shift in range(0, args.angle_samples, args.angle_samples // args.rotation_samples):
            angle = 2 * pi * rot_shift / args.angle_samples

            current_action_image = np.zeros([args.position_samples, args.position_samples])
            current_angle_image = np.zeros([args.position_samples, args.position_samples, args.angle_samples])
            
            for coord_batch, idx_batch in zip(coords_batched, sample_idx_batched):
                model_input_patches = get_patch(image, coord_batch, args.patch_size, angle)
                model_actions, model_angles, _ = model(to_pytorch_img(model_input_patches).cuda())
                model_actions = torch.softmax(model_actions.detach().cpu(), dim=1).numpy()
                model_angles = model_angles.detach().cpu().numpy()

                current_action_image[idx_batch[..., 0], idx_batch[..., 1]] = model_actions[..., 1]
                current_angle_image[idx_batch[..., 0], idx_batch[..., 1]] = np.roll(model_angles, -rot_shift)

                pbar.update()
        
            # plot_inferred(current_action_image, current_angle_image, coords, image)

            action_image += current_action_image
            angle_image += current_action_image[..., None] * current_angle_image    

        action_image = action_image / args.rotation_samples
        angle_image = angle_image / action_image[..., None]

    # np.save("action_image.npy", action_image)
    return action_image, angle_image


def open_image(filename):
    return np.asarray(Image.open(filename)).astype(np.float32) / 255.0




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
    precision = np.mean(np.where(prec_array, 1.0, 0.0))

    rec_array = y_pred[y_true]
    if len(rec_array) == 0:
        return 0.0
    recall = np.mean(np.where(rec_array, 1.0, 0.0))
    if precision == 0 or recall == 0:
        return 0.0
    return (2 * precision * recall / (precision + recall))

def f1_score_fn(y_pred, y_true):
    return f1_score(y_pred > 0.5, y_true > 0.5)

def f1_score_patches_fn(y_pred, y_true):
    return f1_score(patches_pred(y_pred), patches_pred(y_true))




if __name__ == "__main__":
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RoadTracerModel(args.patch_size, input_channels=3, num_angles=args.angle_samples).to(device)
    model.load_state_dict(torch.load(args.load_model))

    image_paths = sorted(glob.glob(f"{args.file_path}/*.png"))
    # mask_paths = sorted(glob.glob(f"{args.file_path}/training/groundtruth/*.png"))

    os.makedirs(args.results_path, exist_ok=True)

    f1_scores = []
    for image_fname in image_paths: #, mask_paths):
        image_base_name = image_fname.split("\\")[-1] 
        actions_result_fname = f"{args.results_path}/{image_base_name.replace('.png', '_actions.npy')}"
        angles_result_fname = f"{args.results_path}/{image_base_name.replace('.png', '_angles.npy')}"

        image = open_image(image_fname)[..., :3]
        
        # assert image_base_name == mask_fname.split("\\")[-1], "Image and mask filenames have to be equal!"
        
        # mask = open_image(mask_fname)
        
        # fix, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(true_patches)
        # ax2.imshow(mask)
        # plt.show()
        if not os.path.exists(actions_result_fname) or not os.path.exists(angles_result_fname):
            pred_actions, pred_angles = inference_immediate(image, model, args)

            np.save(actions_result_fname, pred_actions)
            np.save(angles_result_fname, pred_angles)
        else:
            pred_actions = np.load(actions_result_fname)
            pred_angles = np.load(angles_result_fname)

        


        # true_patches = np.mean(np.reshape(mask, (16, 25, 16, 25)), (1, 3)) > 0.25

        # pred_patches = np.mean(np.reshape(pred, (16, 2, 16, 2)), (1, 3)) > 0.5

        # scores = [f1_score_fn(pred_patches > thr, true_patches) for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]



        # fix, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(true_patches)
        # ax2.imshow(pred_patches)
        # plt.show()

        #score = f1_score_fn(pred_patches, true_patches)
        #f1_scores.append(scores[3])

        #print (f"Current F1 score = {scores}, mean = {np.mean(f1_scores)}")
