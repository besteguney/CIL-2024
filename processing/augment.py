import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import sample
import albumentations as A
import parameters as params

PADDING_WIDTH = 70
IMG_SIZE = 400


additional_transform  = A.Compose([
                        A.RandomRotate90(),
                        A.Flip(),
                        A.OneOf([
                            A.GaussNoise(),
                            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0)),
                            A.MotionBlur(blur_limit=5),
                        ], p=0.2),
                        A.SomeOf([
                            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
                        ], n=2, p=0.8),
                        A.OneOf([
                            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=180, p=0.9),
                            A.RandomResizedCrop(height=IMG_SIZE+2*PADDING_WIDTH, width=IMG_SIZE+2*PADDING_WIDTH, scale=(0.8, 1.0), interpolation=cv2.INTER_LINEAR, p=0.9)
                            ], p=0.8),
                        A.ElasticTransform(alpha=0.5, sigma=25.0, alpha_affine=25.0, p=0.5),
                        A.Crop(x_min=0, y_min=0, x_max=IMG_SIZE+2*PADDING_WIDTH, y_max=IMG_SIZE+2*PADDING_WIDTH, p=1.0)
                    ])

def add_padding(image, mask, pad_width):
    image_padded = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
    mask_padded = np.pad(mask, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=0)
    return image_padded, mask_padded

def remove_padding(image, mask, pad_width):
    image_cropped = image[pad_width:-pad_width, pad_width:-pad_width]
    mask_cropped = mask[pad_width:-pad_width, pad_width:-pad_width]
    return image_cropped, mask_cropped

def augment_data(images, masks, n_times):

    transform = A.Compose([
            A.RandomCrop(width=params.RESIZE, height=params.RESIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, p=0.5),
            A.RandomBrightnessContrast(p=0.4),
    ])

    resize_transform = A.Compose([A.Resize(width=params.RESIZE, height=params.RESIZE, p=1.0)])

    augmented_images = []
    augmented_masks = []
    for img, mask in zip(images, masks):
        resized_result = resize_transform(image = img, mask = mask)
        augmented_images.append(resized_result['image'])
        augmented_masks.append(resized_result['mask'])
        for _ in range(n_times):
            result = transform(image = img, mask = mask)

            augmented_images.append(result['image'])
            augmented_masks.append(result['mask'])

    return augmented_images, augmented_masks