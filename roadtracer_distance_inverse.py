import argparse
import glob
import random

import numpy as np
from matplotlib import pyplot as plt
import torch
import cv2

from roadtracer_utils import get_circle_sample_points, linear_interpolation
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", default="ethz-cil-road-segmentation-2024/training/images", type=str, help="Path to images to run inference for")
parser.add_argument("--ground_truth_path", default="ethz-cil-road-segmentation-2024/training/groundtruth", type=str, help="Path to ground truths")
parser.add_argument("--result_path", default="roadtracer_results_128_test", type=str, help="The folder to find result images in")
parser.add_argument("--run_test", default=True, type=bool, help="Run on the test set - ignore ground truth, save the result as .csv")


parser.add_argument("--step_distance", default=16.0, type=float, help="The length of the edges in the generated graph")
parser.add_argument("--image_size", default=256, type=int, help="The size to which we resize data before training the model")



def linear_interpolation_torch(img, points):
    w, h = img.shape[:2]
    px, py = points[..., 0], points[..., 1]

    x0 = torch.minimum(torch.maximum(torch.tensor(0), px.int()), torch.tensor(w-1))
    y0 = torch.minimum(torch.maximum(torch.tensor(0), py.int()), torch.tensor(h-1))
    x1, y1 = torch.minimum(x0+1, torch.tensor(w-1)), torch.minimum(y0+1, torch.tensor(h-1))
    a, b = px - x0, py - y0
    
    return (1-a) * (1-b) * img[x0, y0] + a * (1-b) * img[x1, y0] + (1-a) * b * img[x0, y1] + a * b * img[x1, y1]


def solve_for_mask(actions, angles, args):
    position_count, _, angle_samples = angles.shape
    
    THRESHOLD = 0.012
    scale_down = 4
    
    coord_row = torch.linspace(0, args.image_size // scale_down, position_count)
    coords = torch.cartesian_prod(coord_row, coord_row) 
    
    distance_image = torch.zeros((args.image_size // scale_down, args.image_size // scale_down), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([distance_image], lr=1)  # Learning rate of 0.1


    # angles_order = torch.tensor(np.argsort(angles, -1).ravel())
    actions = torch.tensor(actions.ravel())
    angles = torch.tensor(angles.ravel())

    step_count = 20
    for it in range(step_count):
        # [image_size**2, 32, 2]
        sample_points = get_circle_sample_points(coords, angle_samples, args.step_distance / scale_down)
        
        pred_dist = linear_interpolation_torch(distance_image, sample_points.reshape(-1, 2))
        target_dist = linear_interpolation_torch(distance_image, coords.repeat_interleave(angle_samples, dim=0)) + args.step_distance * angles # * actions.repeat_interleave(angle_samples)
        loss = torch.mean(actions.repeat_interleave(angle_samples) * torch.square(pred_dist - target_dist))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print (f"Step {it+1: >2d} / {step_count}: Loss = {loss.item()}")
        # if (it + 1) % 5 == 0:
        #     plt.close()
        #     plt.imshow(distance_image.detach().numpy())
        #     plt.draw()
        #     plt.waitforbuttonpress(0.1)
    
    return distance_image.detach().numpy() > THRESHOLD
    
    # plt.imshow(distance_image.detach().numpy())
    # # plt.imshow(distance_image.detach().numpy() > 0.007)
    # #     plt.imshow(distance_image.detach().numpy())
    # plt.draw()
    # plt.waitforbuttonpress(0.1)
    # np.save(f"{args.result_path}/dist_images/{args.image_fname}.npy", distance_image.detach().numpy())



# def solve_for_mask_2(actions, angles, args):
#     w, _, angle_samples = angles.shape
#     result = np.zeros((w, w), dtype=np.float32)
#     for i in range(10000):
#         point = np.random.uniform(0, w, 2)
#         angle=random.uniform(0, 2 * np.pi)
#         velocity = np.array([np.cos(angle), np.sin(angle)])
#         for _ in range(1000):
#             ang = linear_interpolation(angles, point)
#             act = linear_interpolation(actions, point)
#             if random.random() > act:
#                 velocity /= 2
                
#             if np.sum(np.abs(velocity)) < 0.5:
#                 break
                
#             next_point = np.sum(get_circle_sample_points(point, angle_samples, args.step_distance/4) * ang[:, None] / np.sum(ang), 0)
#             velocity = 0.99 * velocity + 0.01 * (next_point - point)
#             point += velocity
#             if point[0] < 0 or point[0] >= w or point[1] < 0 or point[1] >= w:
#                 break
#             result[int(point[0]), int(point[1])] += 1.0
        
#     plt.imshow(result)
#     plt.show()





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



if __name__ == "__main__":
    args = parser.parse_args()
    
    image_nums =  range(144, 288) if args.run_test else range(144)
                
    accuracies = []
    predicted_masks = []
    for i in tqdm(image_nums, ncols=150):       
        actions = np.load(f"{args.result_path}/satimage_{i}_actions.npy")
        angles = np.load(f"{args.result_path}/satimage_{i}_angles.npy")
        
        pred_mask = cv2.resize((solve_for_mask(actions, angles, args)).astype(np.float32), (400, 400), interpolation=cv2.INTER_LINEAR)
        
        if not args.run_test:
            mask_fname = f"{args.ground_truth_path}/satimage_{i}.png"
            reference = np.array(Image.open(mask_fname))
            reference = cv2.resize(reference, (25, 25), interpolation=cv2.INTER_AREA) > 0.25
            
            plt.close()
            _, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(reference)
            ax2.imshow(pred_mask)
            plt.draw()
            plt.waitforbuttonpress(0.1)
        
            acc = [f1_score(pred_mask, reference > 128) for thr in [0.001, 0.002, 0.003, 0.004, 0.007, 0.01, 0.015]]
            accuracies.append(acc)
            print ("Running f-scores: ", [np.mean(x) for x in zip(*accuracies)])
        else:
            predicted_masks.append(pred_mask)
            # plt.imshow(pred_mask)
            # plt.show()
            Image.fromarray((255 * pred_mask).astype(np.uint8)).save(f"{args.result_path}/masks/mask_{i}.png")
            
    
    
    # plt.imshow(distance_image.detach().numpy())
    # # plt.imshow(distance_image.detach().numpy() > 0.007)
    # #     plt.imshow(distance_image.detach().numpy())
    # plt.draw()
    # plt.waitforbuttonpress(0.1)
    
    # main(actions, angles, args)
