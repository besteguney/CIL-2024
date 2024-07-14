import torch
import parameters as params
import segmentation_models_pytorch as smp
import utils
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
        
def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // params.PATCH_SIZE
    w_patches = y.shape[-1] // params.PATCH_SIZE
    patches_hat = y_hat.reshape(-1, 1, h_patches, params.PATCH_SIZE, w_patches, params.PATCH_SIZE).mean((-1, -3)) > params.CUTOFF
    patches = y.reshape(-1, 1, h_patches, params.PATCH_SIZE, w_patches, params.PATCH_SIZE).mean((-1, -3)) > params.CUTOFF
    return (patches == patches_hat).float().mean()

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()
    
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
    # training loop
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': [], 'f1_train': [], 'f1_val': []}
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
            tp, fp, fn, tn = smp.metrics.get_stats(y_hat.long(), y.long(), mode="binary")
            metrics['f1_train'].append(smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise"))
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        if (eval_dataloader == None): 
            continue

        elif (epoch % val_freq == 0) or (epoch==n_epochs):
            # validation
            model.eval()
            with torch.no_grad():  # do not keep track of gradients
                for (x, y) in eval_dataloader:
                    y_hat = model(x)  # forward pass
                    loss = loss_fn(y_hat, y)

                    # log partial metrics
                    metrics['val_loss'].append(loss.item())
                    tp, fp, fn, tn = smp.metrics.get_stats(y_hat.long(), y.long(), mode="binary")
                    metrics['f1_val'].append(smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise"))
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