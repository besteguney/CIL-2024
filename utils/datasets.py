import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .utils import image_to_patches
import matplotlib.pyplot as plt

def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)
'''
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, device, use_patches=True, resize_to=(400, 400)):
        self.use_patches = use_patches
        self.resize_to=resize_to
        self.x = images
        self.y = masks
        self.device = device

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            new_size = (resize_to[0], resize_to[1], 3)
            # preallocate arrays to not resize
            self.x = np.empty((len(images), *new_size), dtype=np.float32)
            self.y = np.empty((len(masks), *resize_to), dtype=np.float32)

            for i in range(len(images)):
                self.x[i] = cv2.resize(images[i], dsize=self.resize_to, interpolation=cv2.INTER_LINEAR)
                self.y[i] = cv2.resize(masks[i], dsize=self.resize_to, interpolation=cv2.INTER_LINEAR)

            # self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            # self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)

        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        return x, y

    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))

    def __len__(self):
        return self.n_samples
    '''



class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, device, use_patches=True, resize_to=(400, 400)):
        self.use_patches = use_patches
        self.resize_to = resize_to
        self.x = images
        self.y = masks
        self.device = device

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        return x, y

    def __getitem__(self, item):
        image = self.x[item]
        mask = self.y[item]
        
        if self.resize_to != (image.shape[1], image.shape[2]):  # resize images
            image = cv2.resize(image, dsize=self.resize_to, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, dsize=self.resize_to, interpolation=cv2.INTER_LINEAR)

        image = np.moveaxis(image, -1, 0)  # convert HWC to CHW format for PyTorch
        mask = np.expand_dims(mask, axis=0)

        return self._preprocess(np_to_tensor(image, self.device), np_to_tensor(mask, self.device))

    def __len__(self):
        return self.n_samples