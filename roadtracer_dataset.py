import torch.utils.data
from torch.utils.data import Dataset
import numpy as np
import utils
import os
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as torchvision_transforms
from roadtracer_graph import graph_step
from roadtracer_utils import linear_interpolation, to_pytorch_img, from_pytorch_img
import random
from math import pi
import matplotlib.pyplot as plt
from roadtracer_utils import RoadTracerImage, get_patch





# class RoadTracerDataset(Dataset):
#     # dataset class that deals with loading the data and making it available by index.

#     def __init__(self, images, masks, device, sample_angles, sample_distance, sample_radius, view_radius):
#         self.images = [RoadTracerImage(img, mask) for img, mask in zip(images, masks)]
#         self.device = device
#         # self.transform = torchvision_transforms.Compose([
#         #     torchvision_transforms.RandomResizedCrop([400, 400], scale=(0.3, 1.0)),
#         #     torchvision_transforms.RandomHorizontalFlip(),
#         # ]) if augmentation else None

#         self.sample_angles = sample_angles
#         self.sample_distance = sample_distance
#         self.sample_radius = sample_radius
#         self.view_radius = view_radius

#     def __getitem__(self, item):
#         image = self.images[item]

#         sample_road = random.random() > 0.5

#         center = random.choice(image.road_samples) if sample_road else random.choice(image.negative_samples)
        
#         sample_points = circle_sample_points(self.sample_angles, self.sample_distance, self.view_radius) + center
#         inputs = to_pytorch_img(linear_interpolation(image.image, sample_points))
        
#         output_scores, output_points = graph_step(image.search, center, self.sample_angles, 1, self.sample_radius)
#         if np.max(output_scores) > 1:
#             print ("Normalization failure, dividing by max")
#             output_scores = output_scores / np.max(output_scores)

#         return utils.np_to_tensor(inputs.astype(np.float32), self.device), utils.np_to_tensor(output_scores.astype(np.float32), self.device) #, linear_interpolation(image.distance, center)

#     def __len__(self):
#         return len(self.images)
        


# def load(images, masks, device, batch_size, angle_samples, distance_samples, sample_radius, view_radius):
#     return torch.utils.data.DataLoader(
#         RoadTracerDataset(images, masks, device, angle_samples, distance_samples, sample_radius, view_radius),
#         batch_size=batch_size,
#         shuffle=True
#     )


# def load_all_data(root_path, device, batch_size, val_size, angle_samples, distance_samples, sample_radius, view_radius):
#     # Loading data
#     images = utils.load_all_from_path(os.path.join(root_path, 'training', 'images'))[:, :, :, :3]
#     masks = utils.load_all_from_path(os.path.join(root_path, 'training', 'groundtruth'))
#     train_images, val_images, train_masks, val_masks = train_test_split(
#         images, masks, test_size=val_size, random_state=42
#     )
#     train = load(train_images, train_masks, device, batch_size, angle_samples, distance_samples, sample_radius, view_radius)
#     val = load(val_images, val_masks, device, batch_size, angle_samples, distance_samples, sample_radius, view_radius)
#     return train, val


class RoadTracerDataset (Dataset):
    def __init__(self, images, patch_size, image_size, positive_samples=None, augmentation=False):
        self.images = images
        self.patch_size = patch_size
        self.image_size = image_size
        self.positive_samples = positive_samples
        self.transform = torchvision_transforms.Compose([
            # torchvision_transforms.RandomResizedCrop([image_size, image_size], scale=(0.3, 1.0)),
            torchvision_transforms.RandomHorizontalFlip(),
            torchvision_transforms.RandomRotation(180)
        ]) if augmentation else None

    def _preprocess(self, *images):
        images = [to_pytorch_img(img) for img in images]
        if self.transform is None:
            return images
        return self.transform(images)

    def _get_random_patch_center(self, item):
        if self.positive_samples is None:
            center = np.random.uniform(0, self.image_size, (2,))
        else:
            sample_road = random.random() < self.positive_samples
            center = random.choice(self.images[item].road_samples) if sample_road else random.choice(self.images[item].negative_samples)
        return center
    
    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        return len(self.images)



class RoadTracerDistanceDataset (RoadTracerDataset):
    def __init__(self, images, patch_size, image_size, positive_samples=0.75, augmentation=False):
        super().__init__(images, patch_size, image_size, positive_samples, augmentation)

    def __getitem__(self, item):
        # TODO scale distance accordingly to image scale?
        center = self._get_random_patch_center(item)
        img, dist = self._preprocess(self.images[item].image, self.images[item].distance)
        return to_pytorch_img(get_patch(from_pytorch_img(img), point, self.patch_size), 2, 0), linear_interpolation(dist, point)


class RoadTracerImmediateDataset (RoadTracerDataset):
    def __init__(self, images, patch_size, image_size, angle_samples, sample_radius, positive_samples=None, augmentation=True):
        super().__init__(images, patch_size, image_size, positive_samples, augmentation)
        self.angle_samples = angle_samples
        self.sample_radius = sample_radius

    def __getitem__(self, item):
        center = self._get_random_patch_center(item)
        img, dist = self._preprocess(self.images[item].image, self.images[item].distance)
        scores, _, _ = graph_step(dist, center, self.angle_samples, self.sample_radius)
        return to_pytorch_img(get_patch(from_pytorch_img(img), center, self.patch_size)).cuda(), torch.from_numpy(scores)


def create_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)



def preprocess_images(images, masks, image_size):
    resize_transform = torchvision_transforms.Resize((image_size, image_size))
    
    resized_images = [resize_transform(img) for img in images]
    resized_masks = [resize_transform(mask) for mask in masks]
    return [RoadTracerImage(img, mask) for img, mask in zip(resized_images, resized_masks)]

def load_all_data(root_path, val_size, image_size):
    # Loading data
    images = utils.load_all_from_path(os.path.join(root_path, 'training', 'images'))[:, :, :, :3]
    masks = utils.load_all_from_path(os.path.join(root_path, 'training', 'groundtruth'))
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=val_size, random_state=42
    )
    train = preprocess_images(train_images, train_masks, image_size)
    val = preprocess_images(val_images, val_masks, image_size)
    return train, val




# TODO specify border behaviour
if __name__ == "__main__":
    data, _ = load_all_data(ROOT_PATH, 0)
    for img, out, dist in train:
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.imshow(img[0].detach().cpu().numpy().astype(np.uint8))
        ax2.imshow(out[0].detach().cpu().numpy()[np.newaxis])
        print (dist)
        plt.show()


