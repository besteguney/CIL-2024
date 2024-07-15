import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import wandb
import numpy as np
from tqdm import tqdm

from models import load_config, model_from_config, loss_fn_from_config
from utils.datasets import ImageDataset
from utils.utils import load_all_from_path, patch_accuracy_fn, accuracy_fn

CHECKPOINT_PATH = Path("checkpoints")
DATA_PATH = Path("data")

# TODO: Somehow add support for other training loops (e.g. step function / multiple train functions)
def train(model, optimizer, train_dl, val_dl, loss_fn, epoch_n, scheduler=None, start_epoch = None, metric_fns = {}, config_name = None, save_interval = sys.maxsize):
    wandb.init(project="CIL",) # I am not sure if you guys can see it.
    model_path = CHECKPOINT_PATH / wandb.run.name
    model_path.mkdir(parents=True, exist_ok=True)

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(start_epoch, start_epoch + epoch_n):
        metrics = {'loss': [], 'val_loss': []}
        
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        train_tqdm = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{start_epoch + epoch_n} Training")

        model.train()

        for (x, y) in train_tqdm:
            optimizer.zero_grad()  # zero out gradients
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            train_tqdm.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        if scheduler != None:
            scheduler.step(loss)
            
         # validation
        model.eval()
        with torch.no_grad():
            val_tqdm = tqdm(val_dl, desc=f"Epoch {epoch + 1}/{start_epoch + epoch_n} Validation")
            for (x, y) in val_tqdm:
                y_hat = model(x)  # forward pass
                loss = loss_fn(y_hat, y)

                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())
                    
                val_tqdm.set_postfix({"val_loss": sum(metrics["val_loss"]) / len(metrics["val_loss"])})
        
        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}

        wandb.log(history[epoch], step=epoch)

        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))

        if (epoch + 1 - start_epoch) % save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': history[epoch]["loss"],
                'wandb_id': wandb.run.id,
                'config_name': config_name
            }, model_path / f"epoch_{epoch + 1}.pt")
        
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': history[epoch]["loss"],
        'wandb_id': wandb.run.id,
        'config_name': config_name
    }, model_path / f"epoch_{epoch + 1}.pt")

    return history

def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training on device: {DEVICE}")

    config = load_config(args.config)

    model = model_from_config(config).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    images = load_all_from_path(str(DATA_PATH / 'training' / 'images'))[:, :, :, :3]
    masks = load_all_from_path(str(DATA_PATH / 'training' / 'groundtruth'))
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    train_ds = ImageDataset(train_images, train_masks, device=DEVICE, use_patches=False, resize_to=(384, 384))
    val_ds = ImageDataset(val_images, val_masks, device=DEVICE, use_patches=False, resize_to=(384, 384))

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)

    if args.continue_from:
        checkpoint = torch.load(CHECKPOINT_PATH / args.continue_from, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optim.param_groups: param_group["lr"] = args.lr
        wandb.init(project="CIL-2024", id=checkpoint['wandb_id'], resume="must")
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from checkpoint {args.continue_from}, starting at epoch {start_epoch}")
    else:
        wandb.init(
            name=args.run_name,
            project="CIL-2024",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "n_training_examples": len(train_ds),
                "n_validation_examples": len(val_ds), 
                "parameter_count": sum([p.numel() for p in model.parameters() if p.requires_grad]),
                **vars(args),
                **config,
            },
            group=config["wandb_group"]
        )
        start_epoch = 0

    loss_fn = loss_fn_from_config(config)
    metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}

    train(model, optim, train_dl, val_dl, loss_fn, args.epochs, start_epoch, metric_fns, args.config, args.save_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNNs/Transformers for Python code generation")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--continue_from", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--save_interval", type=int, default=sys.maxsize, help="Number of epochs between saving model")
    args = parser.parse_args()
    main(args)
