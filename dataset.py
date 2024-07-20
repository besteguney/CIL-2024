import torch.utils.data
from torch.utils.data import Dataset
import cv2
import numpy as np
import utils
import os
from sklearn.model_selection import train_test_split
import torchvision.transforms.v2 as torchvision_transforms


class ImageDataset(Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, images, masks, device, augmentation, use_patches=True, resize_to=None):
        self.device = device
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x, self.y, self.n_samples = images, masks, len(images)
        self._load_data()
        self.transform = torchvision_transforms.Compose([
            torchvision_transforms.RandomResizedCrop([400, 400], scale=(0.3, 1.0)),
            torchvision_transforms.RandomHorizontalFlip(),
            torchvision_transforms.RandomRotation(180)
        ]) if augmentation else None

    def _load_data(self):  # not very scalable, but good enough for now
        if self.use_patches:  # split each image into patches
            self.x, self.y = utils.image_to_patches(self.x, self.y)
        elif self.resize_to is not None and self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        # if self.transform is None:
        # return x, y
        return self.transform([x, y])

    def __getitem__(self, item):
        return self._preprocess(utils.np_to_tensor(self.x[item], self.device), utils.np_to_tensor(self.y[[item]], self.device))

    def __len__(self):
        return self.n_samples


def load(images, masks, device, batch_size, augmentation, use_patches, resize_to=None):
    return torch.utils.data.DataLoader(
        ImageDataset(images, masks, device, augmentation=augmentation, use_patches=use_patches, resize_to=resize_to),
        batch_size=batch_size,
        shuffle=True
    )


def load_all_data(root_path, device, batch_size, use_patches, val_size, resize_to=None):
    # Loading data
    images = utils.load_all_from_path(os.path.join(root_path, 'training', 'images'))[:, :, :, :3]
    masks = utils.load_all_from_path(os.path.join(root_path, 'training', 'groundtruth'))
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=val_size, random_state=42
    )
    train = load(train_images, train_masks, device, batch_size, augmentation=True, use_patches=use_patches, resize_to=resize_to)
    val = load(val_images, val_masks, device, batch_size, augmentation=False, use_patches=use_patches, resize_to=resize_to)
    return train, val

