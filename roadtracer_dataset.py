import torch.utils.data
from torch.utils.data import Dataset
import cv2
import numpy as np
import utils
import os
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as torchvision_transforms
from roadtracer_graph import get_search_image, graph_step, linear_interpolation, circle_sample_points
import random
from parameters import *
from math import pi
import matplotlib.pyplot as plt


def to_pytorch_img(img):
    return np.moveaxis(img, -1, 0)

class RoadTracerImage:
    def __init__(self, img, target):
        self.image = img
        self.distance, self.search = get_search_image((255*target).astype(np.uint8))
        self.target = target

        self.road_samples = np.stack(np.nonzero(self.target != 0), 1)
        self.negative_samples = np.stack(np.nonzero(self.target == 0), 1)


class RoadTracerDataset(Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, images, masks, device, sample_angles, sample_distance, sample_radius, view_radius):
        self.images = [RoadTracerImage(img, mask) for img, mask in zip(images, masks)]
        self.device = device
        # self.transform = torchvision_transforms.Compose([
        #     torchvision_transforms.RandomResizedCrop([400, 400], scale=(0.3, 1.0)),
        #     torchvision_transforms.RandomHorizontalFlip(),
        # ]) if augmentation else None

        self.sample_angles = sample_angles
        self.sample_distance = sample_distance
        self.sample_radius = sample_radius
        self.view_radius = view_radius

    def __getitem__(self, item):
        image = self.images[item]

        sample_road = random.random() > 0.5

        center = random.choice(image.road_samples) if sample_road else random.choice(image.negative_samples)
        
        sample_points = circle_sample_points(self.sample_angles, self.sample_distance, self.view_radius) + center
        inputs = to_pytorch_img(linear_interpolation(image.image, sample_points))
        
        output_scores, output_points = graph_step(image.search, center, self.sample_angles, 1, self.sample_radius)

        return utils.np_to_tensor(inputs.astype(np.float32), self.device), utils.np_to_tensor(output_scores.astype(np.float32), self.device) #, linear_interpolation(image.distance, center)

    def __len__(self):
        return len(self.images)
        


def load(images, masks, device, batch_size, angle_samples, distance_samples, sample_radius, view_radius):
    return torch.utils.data.DataLoader(
        RoadTracerDataset(images, masks, device, angle_samples, distance_samples, sample_radius, view_radius),
        batch_size=batch_size,
        shuffle=True
    )


def load_all_data(root_path, device, batch_size, val_size, angle_samples, distance_samples, sample_radius, view_radius):
    # Loading data
    images = utils.load_all_from_path(os.path.join(root_path, 'training', 'images'))[:, :, :, :3]
    masks = utils.load_all_from_path(os.path.join(root_path, 'training', 'groundtruth'))
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=val_size, random_state=42
    )
    train = load(train_images, train_masks, device, batch_size, angle_samples, distance_samples, sample_radius, view_radius)
    val = load(val_images, val_masks, device, batch_size, angle_samples, distance_samples, sample_radius, view_radius)
    return train, val



# TODO specify border behaviour
if __name__ == "__main__":
    train, val = load_all_data(ROOT_PATH, "cuda", 1, 24, 200, 60, 15, 30)
    for img, out, dist in train:
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.imshow(img[0].detach().cpu().numpy().astype(np.uint8))
        ax2.imshow(out[0].detach().cpu().numpy()[np.newaxis])
        print (dist)
        plt.show()


