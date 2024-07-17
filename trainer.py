import torch
import parameters as params
import segmentation_models_pytorch as smp
import utils
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm


def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, val_freq):
    # training loop
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
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

            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        if (eval_dataloader == None): 
            continue

        if (epoch % val_freq == 0) or (epoch==n_epochs):
            # validation
            model.eval()
            with torch.no_grad():  # do not keep track of gradients
                for (x, y) in eval_dataloader:
                    y_hat = model(x)  # forward pass
                    loss = loss_fn(y_hat, y)

                    # log partial metrics
                    metrics['val_loss'].append(loss.item())
                    for k, fn in metric_fns.items():
                        metrics['val_'+k].append(fn(y_hat, y).item())

            # summarize metrics, log to tensorboard and display
            history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
            for k, v in history[epoch].items():
                writer.add_scalar(k, v, epoch)
            print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
            utils.show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

    print('Finished Training')
    # plot loss curves
    plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
    plt.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def train_smp(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, val_freq):
    # Initialize tensorboard writer
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)

    history = {}  # Collects metrics at the end of each epoch
    device = next(model.parameters()).device
    for epoch in range(n_epochs):  # Loop over the dataset multiple times

        # Initialize metric list
        metrics = {
            'loss': [], 'val_loss': [], 
            'f1_train': [], 'f1_val': [], 
            'acc_train': [], 'acc_val': []
        }
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_' + k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        
        # Training phase
        model.train()
        for (x, y) in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()  # Zero out gradients
            y_hat = model(x)  # Forward pass
            loss = loss_fn(y_hat, y)
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights

            # Log partial metrics
            metrics['loss'].append(loss.item())
            
            predictions = to_preds(y_hat)
            # Calculate F1 score
            tp, fp, fn, tn = smp.metrics.get_stats(predictions.long(), y.long(), mode="binary")
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise").item()
            metrics['f1_train'].append(f1_score)
            
            # Calculate accuracy
            correct = (predictions == y).float().sum()
            acc = correct / y.numel()
            metrics['acc_train'].append(acc.item())
            
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        if eval_dataloader and ((epoch % val_freq == 0) or (epoch == n_epochs - 1)):
            # Validation phase
            model.eval()
            with torch.no_grad():  # Do not keep track of gradients
                for (x, y) in eval_dataloader:
                    x, y = x.to(device), y.to(device)
                    y_hat = model(x)  # Forward pass
                    loss = loss_fn(y_hat, y)

                    # Log partial metrics
                    metrics['val_loss'].append(loss.item())

                    predictions = to_preds(y_hat)

                    tp, fp, fn, tn = smp.metrics.get_stats(predictions.long(), y.long(), mode="binary")
                    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise").item()
                    metrics['f1_val'].append(f1_score)

                    # Calculate accuracy
                    correct = (predictions == y).float().sum()
                    acc = correct / y.numel()
                    metrics['acc_val'].append(acc.item())

                    for k, fn in metric_fns.items():
                        metrics['val_' + k].append(fn(y_hat, y).item())

            # Summarize metrics, log to tensorboard and display
            history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
            for k, v in history[epoch].items():
                writer.add_scalar(k, v, epoch)
            print(' '.join(['\t- ' + str(k) + ' = ' + str(v) + '\n ' for (k, v) in history[epoch].items()]))
            utils.show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

    print('Finished Training')
    
    # Plot loss curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
    plt.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # Plot F1 score curves
    plt.subplot(1, 3, 2)
    plt.plot([v['f1_train'] for k, v in history.items()], label='Training F1 Score')
    plt.plot([v['f1_val'] for k, v in history.items()], label='Validation F1 Score')
    plt.ylabel('F1 Score')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy curves
    plt.subplot(1, 3, 3)
    plt.plot([v['acc_train'] for k, v in history.items()], label='Training Accuracy')
    plt.plot([v['acc_val'] for k, v in history.items()], label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_pix2pix(train_dataloader, eval_dataloader, generator, discriminator, g_loss, d_loss, metric_fns, g_optimizer, d_optimizer, n_epochs):
    # training loop
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'g_loss': [], 'd_loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')

        # Training
        generator.train()
        discriminator.train()
        for (x,y) in pbar:
            # Generator
            fake_image = generator(x)
            fake_pred = discriminator(fake_image, x)
            generator_loss = g_loss(fake_image, y, fake_pred)

            # Discriminator
            fake_image = generator(x).detach()
            fake_pred = discriminator(fake_image, x)
            real_pred = discriminator(y, x)
            discriminator_loss = d_loss(fake_pred, real_pred)


            # Performing the parameter updates
            g_optimizer.zero_grad()
            generator_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            discriminator_loss.backward()
            d_optimizer.step()

            metrics['g_loss'].append(generator_loss.item())
            metrics['d_loss'].append(discriminator_loss.item())

            for k, fn in metric_fns.items():
                metrics[k].append(fn(fake_image, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        if (epoch + 1) % 20 == 0:
            checkpoint_path = f'models/checkpoint_epoch_generator_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
                'loss': g_loss,
            }, checkpoint_path)
            checkpoint_path = f'models/checkpoint_epoch_discriminator_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': discriminator.state_dict(),
                'optimizer_state_dict': d_optimizer.state_dict(),
                'loss': d_loss,
            }, checkpoint_path)


        # validation
        generator.eval()
        discriminator.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y_hat = generator(x)  # forward pass
                fake_pred = discriminator(y_hat, x)
                loss = g_loss(y_hat, y, fake_pred)

                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
          writer.add_scalar(k, v, epoch)
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
        utils.show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

    print('Finished Training')
    # plot loss curves
    plt.plot([v['d_loss'] for k, v in history.items()], label='Discriminator Loss')
    plt.plot([v['g_loss'] for k, v in history.items()], label='Generator Loss')
    plt.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()