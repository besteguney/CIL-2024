import numpy as np
import torch
import cv2
from glob import glob
from PIL import Image
from torchvision import transforms

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
        self.resize_to=resize_to
        self.x = images
        self.y = masks
        self.device = device

        if self.use_patches:  # split each image into patches
            self.x, self.y = image_to_patches(self.x, self.y)
        elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)

    def _preprocess(self, x, y):
        return x, y

    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))

    def __len__(self):
        return self.n_samples


def tensor_to_device(x: torch.Tensor, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return x.cpu()
    else:
        return x.contiguous().pin_memory().to(device=device, non_blocking=True)


class LargeImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, device, resize_to=(400, 400)):
        self.resize_to=resize_to
        self.image_filenames = sorted(glob(image_dir + '/*.png'))
        self.mask_filenames = sorted(glob(mask_dir + '/*.png'))
        self.device = device

        assert len(self.image_filenames) == len(self.mask_filenames)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_to),
        ])

        self.n_samples = len(self.image_filenames)

    def _preprocess(self, x, y):
        return x, y

    def __getitem__(self, item):
        return tensor_to_device(self.transforms(Image.open(self.image_filenames[item])), self.device), tensor_to_device(self.transforms(Image.open(self.mask_filenames[item])), self.device)

    def __len__(self):
        return self.n_samples