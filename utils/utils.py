import math
import os
import re
import cv2
import numpy as np
import utils
import torch
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import Image
from torch import nn
import parameters as params
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import segmentation_models_pytorch as smp


# Should this go somewhere else?
PATCH_SIZE = 16
CUTOFF = 0.5
ROOT_PATH = "./"


def load_all_from_path(path):
    # loads all HxW .pngs contained in path as a 4D np.array of shape (n_images, H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([np.array(Image.open(f)) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.0

def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)
        

def show_first_n(imgs, masks, n=5):
    # visualizes the first n elements of a series of images and segmentation masks
    imgs_to_draw = min(5, len(imgs))
    fig, axs = plt.subplots(2, imgs_to_draw, figsize=(18.5, 6))
    for i in range(imgs_to_draw):
        axs[0, i].imshow(imgs[i])
        axs[1, i].imshow(masks[i])
        axs[0, i].set_title(f'Image {i}')
        axs[1, i].set_title(f'Mask {i}')
        axs[0, i].set_axis_off()
        axs[1, i].set_axis_off()
    plt.show()

def image_to_patches(images, masks=None):
    # takes in a 4D np.array containing images and (optionally) a 4D np.array containing the segmentation masks
    # returns a 4D np.array with an ordered sequence of patches extracted from the image and (optionally) a np.array containing labels
    n_images = images.shape[0]  # number of images
    h, w = images.shape[1:3]  # shape of images
    assert (h % params.PATCH_SIZE) + (w % params.PATCH_SIZE) == 0  # make sure images can be patched exactly

    images = images[:,:,:,:3]

    h_patches = h // params.PATCH_SIZE
    w_patches = w // params.PATCH_SIZE

    patches = images.reshape((n_images, h_patches, params.PATCH_SIZE, w_patches, params.PATCH_SIZE, -1))
    patches = np.moveaxis(patches, 2, 3)
    patches = patches.reshape(-1, params.PATCH_SIZE, params.PATCH_SIZE, 3)
    if masks is None:
        return patches

    masks = masks.reshape((n_images, h_patches, params.PATCH_SIZE, w_patches, params.PATCH_SIZE, -1))
    masks = np.moveaxis(masks, 2, 3)
    labels = np.mean(masks, (-1, -2, -3)) > params.CUTOFF  # compute labels
    labels = labels.reshape(-1).astype(np.float32)
    return patches, labels


def show_patched_image(patches, labels, h_patches=25, w_patches=25):
    # reorders a set of patches in their original 2D shape and visualizes them
    fig, axs = plt.subplots(h_patches, w_patches, figsize=(18.5, 18.5))
    for i, (p, l) in enumerate(zip(patches, labels)):
        # the np.maximum operation paints patches labeled as road red
        axs[i // w_patches, i % w_patches].imshow(np.maximum(p, np.array([l.item(), 0., 0.])))
        axs[i // w_patches, i % w_patches].set_axis_off()
    plt.show()

def show_val_samples(x, y, y_hat, segmentation=False):
    # training callback to show predictions on validation set
    imgs_to_draw = min(5, len(x))
    if x.shape[-2:] == y.shape[-2:]:  # segmentation
        fig, axs = plt.subplots(3, imgs_to_draw, figsize=(18.5, 12))
        if imgs_to_draw > 1:
            for i in range(imgs_to_draw):
                axs[0, i].imshow(np.moveaxis(x[i], 0, -1))
                axs[1, i].imshow(np.concatenate([np.moveaxis(y_hat[i], 0, -1)] * 3, -1))
                axs[2, i].imshow(np.concatenate([np.moveaxis(y[i], 0, -1)]*3, -1))
                axs[0, i].set_title(f'Sample {i}')
                axs[1, i].set_title(f'Predicted {i}')
                axs[2, i].set_title(f'True {i}')
                axs[0, i].set_axis_off()
                axs[1, i].set_axis_off()
                axs[2, i].set_axis_off()
        else:
            axs[0].imshow(np.moveaxis(x[0], 0, -1))
            axs[1].imshow(np.concatenate([np.moveaxis(y_hat[0], 0, -1)] * 3, -1))
            axs[2].imshow(np.concatenate([np.moveaxis(y[0], 0, -1)]*3, -1))
            axs[0].set_title(f'Sample {0}')
            axs[1].set_title(f'Predicted {0}')
            axs[2].set_title(f'True {0}')
            axs[0].set_axis_off()
            axs[1].set_axis_off()
            axs[2].set_axis_off()
    else:  # classification
        fig, axs = plt.subplots(1, imgs_to_draw, figsize=(18.5, 6))
        for i in range(imgs_to_draw):
            axs[i].imshow(np.moveaxis(x[i], 0, -1))
            axs[i].set_title(f'True: {np.round(y[i]).item()}; Predicted: {np.round(y_hat[i]).item()}')
            axs[i].set_axis_off()
    plt.show()

def load_the_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def create_submission(test_folder, test_subfolder, submission_filename, model, device, resize=params.RESIZE):
    test_path = os.path.join(params.ROOT_PATH, test_folder, test_subfolder)
    test_filenames = (glob(test_path + '/*.png'))
    test_images = load_all_from_path(test_path)
    batch_size = test_images.shape[0]
    size = test_images.shape[1:3]

    test_images = np.stack([cv2.resize(img, dsize=(resize, resize)) for img in test_images], 0)
    test_images = test_images[:, :, :, :3]
    test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
    test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
    test_pred = np.concatenate(test_pred, 0)
    test_pred= np.moveaxis(test_pred, 1, -1)  # CHW to HWC
    test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape
    # now compute labels
    test_pred = test_pred.reshape((-1, size[0] // params.PATCH_SIZE, params.PATCH_SIZE, size[0] // params.PATCH_SIZE, params.PATCH_SIZE))
    test_pred = np.moveaxis(test_pred, 2, 3)
    test_pred = np.round(np.mean(test_pred, (-1, -2)) > params.CUTOFF)

    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), test_pred):
            img_number = int(re.search(r"satimage_(\d+)", fn).group(1))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j*params.PATCH_SIZE, i*params.PATCH_SIZE, int(patch_array[i, j])))


def load_images(image_folder_path, is_label = False):
    images = []
    for filename in os.listdir(image_folder_path):
        img_path = os.path.join(image_folder_path, filename)
        img = Image.open(img_path)
        if (is_label):
            img = img.convert('L')
        elif img.mode == 'RGBA':
            img = img.convert('RGB')
        images.append(np.array(img))
    return images

def show_image(image_array, mask_array):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image_array)
    ax[0].set_title('Image')
    ax[0].axis('off')

    ax[1].imshow(mask_array, cmap='gray')
    ax[1].set_title('Mask')
    ax[1].axis('off')

    plt.show()

def overlay_image(image_array, mask_array):
    plt.figure(figsize=(6, 6))
    plt.imshow(image_array)
    plt.imshow(mask_array, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

def to_preds(logits):
    probs = torch.sigmoid(logits)
    preds = (probs >= params.CUTOFF).float()
    return preds

def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // params.PATCH_SIZE
    w_patches = y.shape[-1] // params.PATCH_SIZE
    patches_hat = y_hat.reshape(-1, 1, h_patches, params.PATCH_SIZE, w_patches, params.PATCH_SIZE).mean((-1, -3)) > params.CUTOFF
    patches = y.reshape(-1, 1, h_patches, params.PATCH_SIZE, w_patches, params.PATCH_SIZE).mean((-1, -3)) > params.CUTOFF
    return (patches == patches_hat).float().mean()

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    preds = to_preds(y_hat)
    return (preds == y.round()).float().mean()

def f1_fn(y_hat, y):
    preds = to_preds(y_hat)
    tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), y.long(), mode="binary")
    return smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

def ensemble_predict(models, weights, test_images):
    if len(models) != len(weights):
        raise ValueError("Number of models and weights must be the same")
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    predictions = []
    
    for model, weight in zip(models, weights):
        # Assuming each model has a predict method
        test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
        test_pred = np.concatenate(test_pred, 0)
        print(test_pred.shape)
        test_pred= np.moveaxis(test_pred, 1, -1)
        predictions.append(test_pred)

    weighted_sum_tensor = np.zeros_like(predictions[0])
    for tensor, weight in zip(predictions, weights):
        weighted_sum_tensor += weight * tensor
    return weighted_sum_tensor
