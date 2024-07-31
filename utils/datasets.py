import numpy as np
import torch
import cv2
from .utils import image_to_patches

def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)

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