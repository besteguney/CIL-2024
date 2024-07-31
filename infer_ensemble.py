import numpy as np
import re
import os
import cv2
from glob import glob
import segmentation_models_pytorch as smp
import torch
from utils.utils import load_all_from_path, np_to_tensor, to_preds
import parameters as params

def get_unique_name(base, directory):
    """
    Given a filepath, append a number to the end if the file already exists.
    """
    extension = '.pth'
    counter = 1
    filename = f"{base}_{counter}{extension}"
    while os.path.isfile(os.path.join(directory,filename)):
        filename = f"{base}_{counter}{extension}"
        counter += 1
    return filename[:-4]

def generate_filename(encoder_name, architecture="Unet", img_size=256):
    return '_'.join([architecture, encoder_name, str(img_size)])

def model_init(encoder_name, architecture="Unet"):
    if architecture == "Unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1
        )
    elif architecture == "UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1
        )
    elif architecture == "FPN":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            in_channels=3,
            classes=1,
            upsampling=4
        )
    elif architecture == "PSPNet":
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            in_channels=3,
            classes=1,
            upsampling=8
        )
    elif architecture == "PAN":
        model = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            decoder_channels=32,
            in_channels=3,
            classes=1
        )
    elif architecture == "Linknet":
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            in_channels=3,
            classes=1
        )
    elif architecture == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=3,
            classes=1,
            upsampling=4
        )
    elif architecture == "EfficientUnetPlusPlus":
        model = smp.EfficientUnetPlusPlus(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1
        )
    elif architecture == "ResUnet":
        model = smp.ResUnet(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1
        )
    elif architecture == "ResUnetPlusPlus":
        model = smp.ResUnetPlusPlus(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            in_channels=3,
            classes=1
        )
    elif architecture == "MAnet":
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_pab_channels=64,
            in_channels=3,
            classes=1
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model

def create_ensemble_submission(test_folder, test_subfolder, submission_filename, models_path, device):
    preds = create_ensemble_preds(test_folder, test_subfolder, models_path, device)
    create_submission_file(preds, test_folder, test_subfolder, submission_filename)


def create_submission_file(final_pred, test_folder, test_subfolder, submission_filename):
    test_path = os.path.join(params.ROOT_PATH2, test_folder, test_subfolder)
    test_filenames = (glob(test_path + '/*.png'))
    test_data = load_all_from_path(test_path)
    size = test_data.shape[1:3]

    # produce patches
    final_pred = final_pred.reshape((-1, size[0] // params.PATCH_SIZE, params.PATCH_SIZE, size[0] // params.PATCH_SIZE, params.PATCH_SIZE))
    final_pred = np.moveaxis(final_pred, 2, 3)
    final_pred = np.round(np.mean(final_pred, (-1, -2)) > params.CUTOFF)
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), final_pred):
            img_number = int(re.search(r"satimage_(\d+)", fn).group(1))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j * params.PATCH_SIZE, i * params.PATCH_SIZE, int(patch_array[i, j])))


def create_ensemble_preds(test_folder, test_subfolder, models_path, device):
    test_path = os.path.join(params.ROOT_PATH2, test_folder, test_subfolder)
    test_data = load_all_from_path(test_path)
    size = test_data.shape[1:3]
    predictions = []
    model_dir = os.listdir(models_path)
    weights = []

    for filename in model_dir:
        # get configs for the trained model
        configs = filename.split('_')
        architecture = configs[0]
        encoder_name = configs[1]
        num_pixels = int(configs[2])
        resize_to = (num_pixels, num_pixels)

        test_images = np.stack([cv2.resize(img, dsize=resize_to) for img in test_data], 0)
        test_images = test_images[:, :, :, :3]
        test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
        model = model_init(encoder_name, architecture)

        model_path = os.path.join(models_path, filename)


        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        
        # make predictions
        test_pred = [torch.sigmoid(model(t)).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
        test_pred = np.concatenate(test_pred, 0)
        test_pred = np.moveaxis(test_pred, 1, -1)
        test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)
        predictions.append(test_pred)

        # determine weight based on num_pixels
        if num_pixels == 256:
            weight = 1.0  # less weight
        elif num_pixels == 384:
            weight = 1.3  # more weight
        else:
            weight = 1.0  # default weight for other sizes
        weights.append(weight)

    # weighted average predictions
    sum_prediction = np.zeros_like(predictions[0])
    total_weight = sum(weights)
    for pred, weight in zip(predictions, weights):
        sum_prediction += pred * weight
    final_pred = sum_prediction / total_weight

    return final_pred
