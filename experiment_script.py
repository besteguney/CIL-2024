import argparse
from pathlib import Path
import sys
import os

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torchmetrics import F1Score

import wandb
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
import segmentation_models_pytorch as smp


from models import load_config, model_from_config, loss_fn_from_config
from utils.datasets import ImageDataset
from utils.utils import load_all_from_path, patch_accuracy_fn, accuracy_fn, to_preds, patch_f1_fn
import parameters as params

RANDOM_STATES = [17,89,79]
EPOCHS = 50
LR = 1e-4

def get_extra_data(folders):
    folder_sizes = [len(glob(folder + '/*_label.png')) for folder in folders]
    images = np.stack([np.array(Image.open(folders[j] + f'/{i+1}.png'))[:,:,:3] for j in range(len(folders)) for i in range(folder_sizes[j])]).astype(np.float32) / 255.0
    graphs = np.stack([np.array(Image.open(folders[j] + f'/{i+1}_pred.png')) for j in range(len(folders)) for i in range(folder_sizes[j])]).astype(np.float32) / 255.0
    masks = np.stack([np.array(Image.open(folders[j] + f'/{i+1}_label.png'))[:,:,0] for j in range(len(folders)) for i in range(folder_sizes[j])]).astype(np.float32) / 255.0
    return images, graphs, masks

def train_smp_wandb(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, val_freq=10, wandb_run=None, save_location=None):
    min_loss = float('inf')
    history = {}  # collects metrics at the end of each epoch
    f1_metric = F1Score(task='binary', num_classes=2, average='macro').to(next(model.parameters()).device)
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {
            'loss': [], 'val_loss': [], 
            'f1_train': [], 'f1_val': [], 
            'acc_train': [], 'acc_val': []
        }
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        # training
        model.train()
        for (x, y) in pbar:
            optimizer.zero_grad()  # zero out gradients
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            metrics['loss'].append(loss.item()) 
            predictions = to_preds(y_hat)
            # calculate f1 score
            f1_score_value = f1_metric(predictions.long(), y.long())
            metrics['f1_train'].append(f1_score_value.item())

            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        if eval_dataloader and ((epoch % val_freq == 0) or (epoch == n_epochs - 1)):
            # validation
            model.eval()
            with torch.no_grad():  # do not keep track of gradients
                for (x, y) in eval_dataloader:
                    y_hat = model(x)  # forward pass
                    loss = loss_fn(y_hat, y)
                    metrics['val_loss'].append(loss.item())
                    
                    predictions = to_preds(y_hat)
                    # calculate f1 score
                    f1_score_value = f1_metric(predictions.long(), y.long())
                    metrics['f1_val'].append(f1_score_value.item())

                    for k, fn in metric_fns.items():
                        metrics['val_'+k].append(fn(y_hat, y).item())

            # summarize metrics, log to W&B and display
            history[epoch] = {k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0}
            if wandb_run:
                wandb_run.log(history[epoch], step=epoch)
            if save_location and history[epoch]['val_loss'] < min_loss:
                min_loss = history[epoch]['val_loss']
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': history[epoch]["loss"],
                    'wandb_id': wandb.run.id,
                }, f"checkpoints/{save_location}/checkpoints.pt")
            print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))

    wandb_run.finish()
    print('Finished Training')

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    images = load_all_from_path(params.ROOT_PATH + '/training/images')[:, :, :, :3]
    masks = load_all_from_path(params.ROOT_PATH + '/training/groundtruth')
    graphs = load_all_from_path(params.ROOT_PATH + '/training/roadtracer')

    extra_folders = [params.ROOT_PATH + f'/cil_data/{i}_ZOOM_18' for i in range(6)]
    extra_images, extra_graphs, extra_masks = get_extra_data(extra_folders)

    images = np.concatenate([images, extra_images], axis=0)
    graphs = np.concatenate([graphs, extra_graphs], axis=0)
    masks = np.concatenate([masks, extra_masks], axis=0)

    if args.graph:
        images_with_graphs = np.concatenate([images, graphs[:,:,:,None]], axis=-1)

    for i in range(3):
        if args.graph:
            train_images, val_images, train_masks, val_masks = train_test_split(
                images_with_graphs, masks, test_size=0.1, random_state=RANDOM_STATES[i]
            )
        else:
            train_images, val_images, train_masks, val_masks = train_test_split(
                images, masks, test_size=0.1, random_state=RANDOM_STATES[i]
            )


        train_ds = ImageDataset(train_images, train_masks, device=DEVICE, use_patches=False, resize_to=(384, 384))
        val_ds = ImageDataset(val_images, val_masks, device=DEVICE, use_patches=False, resize_to=(384, 384))

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=True)

        if args.model == 'resunet':
            model_graph = smp.ResUnet(
                encoder_name=args.encoder,
                encoder_depth=5,
                encoder_weights='imagenet',
                decoder_channels=(256, 128, 64, 32, 16),
                in_channels=train_images.shape[-1],
                classes=1,
            ).to(DEVICE)
        else:
            raise Exception(f'not supported model argument: {args.model}')


        loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        optimizer_graph = torch.optim.Adam(model_graph.parameters(), lr=1e-4)

        metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn, 'patch_f1': patch_f1_fn}

        wandb_run = wandb.init(
            name = f"{RANDOM_STATES[i]} {'with_graph' if args.graph else 'without_graph'}",
            project="CIL-experiments",
            config={
                "learning_rate": LR,
                "epochs": EPOCHS,
                "n_training_examples": len(train_ds),
                "n_validation_examples": len(val_ds), 
                "parameter_count": sum([p.numel() for p in model_graph.parameters() if p.requires_grad]),
            },
            group=f'{args.model} {args.encoder}'
        )

        print(f"Training on device: {DEVICE} with random state {RANDOM_STATES[i]}")

        # save_folder = f'checkpoints/{args.model}_{args.encoder}_{RANDOM_STATES[i]}_{args.graph}/'
        # os.mkdir(save_folder)

        train_smp_wandb(
            train_dataloader=train_dl,
            eval_dataloader=val_dl,
            model=model_graph,
            loss_fn=loss_fn,
            metric_fns=metric_fns,
            optimizer=optimizer_graph,
            n_epochs=EPOCHS,
            val_freq=1,
            wandb_run=wandb_run,
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="model type (resunet ...)")
    parser.add_argument("encoder", type=str, help="encoder type (smp supported)")
    parser.add_argument("-graph", action='store_true', help="use graph")
    args = parser.parse_args()
    main(args)
