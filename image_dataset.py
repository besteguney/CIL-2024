import torch
import utils
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by index.

    def __init__(self, images, masks, device, use_patches=True, resize_to=(400, 400)):
        self.device = device
        self.use_patches = use_patches
        self.resize_to=resize_to
        self.x = images
        self.y = masks
        self.n_samples = images.shape[0]
        self._load_data()

    def _load_data(self):  # not very scalable, but good enough for now

        if self.use_patches:  # split each image into patches
            self.x, self.y = utils.image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        return x, y

    def __getitem__(self, item):
        return self._preprocess(utils.np_to_tensor(self.x[item], self.device), utils.np_to_tensor(self.y[[item]], self.device))

    def __len__(self):
        return self.n_samples