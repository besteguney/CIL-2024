import os
import re
import cv2
import torch
import numpy as np
import parameters as params
from utils import utils
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import Image
from torch import nn
import trainer
from sklearn.model_selection import train_test_split
from utils.datasets import ImageDataset
from utils.losses import DiceBCELoss
import random
import segmentation_models_pytorch as smp
import argparse
import sys
import wandb
from utils.utils import load_all_from_path, patch_accuracy_fn, accuracy_fn
from infer_ensemble import model_init, get_unique_name, generate_filename


# For original dataset
def load_images(image_folder_path, is_label = False):
    images = []
    for filename in os.listdir(image_folder_path):
        img_path = os.path.join(image_folder_path, filename)
        img = Image.open(img_path)
        if (is_label):
            img = img.convert('L')
        elif img.mode == 'RGBA':
            img = img.convert('RGB')
        images.append(np.array(img).astype(np.float32) / 255.0)
    return images

def load_scraped_images(location_id):
    data_path = "C:\\Users\\3mmyz\\Documents\\cil_data"
    folder = str(location_id) + '_ZOOM_18'
    image_folder_path = os.path.join(data_path, folder)

    images = []
    masks = []

    files = os.listdir(image_folder_path)
    total_samples = int(len(files) / 2)
    num_samples = int(total_samples * params.LOCATIONS[location_id])
    sampled_indices = random.sample(list(range(total_samples)), num_samples)
    for i in sampled_indices:
        img_path = os.path.join(image_folder_path, str(i+1) + '.png')
        mask_path = os.path.join(image_folder_path, str(i+1) + '_label.png')
        img = Image.open(img_path)
        img = img.convert('RGB')
        
        mask = Image.open(mask_path)
        mask = mask.convert('L')

        images.append(np.array(img).astype(np.float32) / 255.0)
        masks.append(np.array(mask).astype(np.float32) / 255.0)
    return images, masks

def load_extra_data():
    all_images = []
    all_masks = []
    for i in range(24):
        print("Collecting data from location " + str(i))
        images, masks = load_scraped_images(i)

        all_images.extend(images)
        all_masks.extend(masks)
    return all_images, all_masks

def get_data():
    images_list = load_images(os.path.join('ethz-cil-road-segmentation-2024', 'training', 'images'), False)
    masks_list = load_images(os.path.join('ethz-cil-road-segmentation-2024', 'training', 'groundtruth'), True)

    images_extra, masks_extra = load_extra_data()

    images_list.extend(images_extra)
    masks_list.extend(masks_extra)

    num_images = len(images_list)
    image_shape = images_list[0].shape
    mask_shape = masks_list[0].shape

    # preallocate arrays to not resize
    images = np.empty((num_images, *image_shape), dtype=np.float32)
    masks = np.empty((num_images, *mask_shape), dtype=np.float32)

    # fill the preallocated arrays
    for i in range(num_images):
        images[i] = images_list[i]
        masks[i] = masks_list[i]

    return images, masks


def main(encoder):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training Unet using encoder: " + args.encoder)

    images, masks = get_data()

    print("Finished data collection.")

    train_images, val_images, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.1, random_state=42, shuffle=True
    )


    # reshape the image to simplify the handling of skip connections and maxpooling
    train_dataset = ImageDataset(train_images, train_masks, device, use_patches=False, resize_to=(args.size, args.size))
    val_dataset = ImageDataset(val_images, val_masks, device, use_patches=False, resize_to=(args.size, args.size))
    full_dataset = ImageDataset(images, masks, device, use_patches=False, resize_to=(args.size, args.size))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params.BATCH_SIZE, shuffle=True)
    full_dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=params.BATCH_SIZE, shuffle=True)

    model = model_init(args.encoder, args.architecture)
    
    model = model.to(device)
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


    exp_name = generate_filename(args.encoder, args.architecture, args.size)
    save_name = get_unique_name(exp_name, params.SAVED_MODELS_PATH)

    wandb_run = wandb.init(project="cil", entity="emmy-zhou", name=save_name)
    trainer.train_smp_wandb(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, 50, 1, wandb_run)
    torch.save(model.state_dict(), os.path.join(params.SAVED_MODELS_PATH, save_name + '.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ensemble model")
    parser.add_argument("--architecture", type=str, default="Unet")
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()
    main(args)
