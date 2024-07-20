import torch
import parameters as params
import segmentation_models_pytorch as smp
import utils
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
from roadtracer_graph import BaseNode, Node, RoadTracerImage, get_oracle_prediction
import numpy as np
from roadtracer_utils import get_circle_sample_points
import cv2


def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs):
    # training loop
    logdir = f'./tensorboard/roadtracer-{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        # training
        model.train()
        for (x, y) in tqdm(train_dataloader, ncols=150, desc=f"Epoch {epoch+1}/{n_epochs}"):
            optimizer.zero_grad()  # zero out gradients
            y_hat = model(x)  # forward pass
            loss = loss_fn(y_hat, y)
            loss.backward()  # backward pass
            optimizer.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y))
            # pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        if eval_dataloader is None:
            continue

        # validation
        model.eval()
        first = True
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y_hat = model(x)  # forward pass
                loss = loss_fn(y_hat, y)

                if first:
                    writer.add_images("input", x, epoch)
                    writer.add_images("output", y_hat[:, None, None], epoch)
                    writer.add_images("ground truth", y[:, None, None], epoch)
                    first = False

                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y))

        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
            writer.add_scalar(k, v, epoch)
        
        writer.flush()
        # print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
        # utils.show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

    print('Finished Training')
    # plot loss curves
    plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
    plt.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()





def get_patch(image, point, patch_size):
    step = (patch_size-1)/2
    coords = np.linspace(point-step, point+step, patch_size)
    grid = np.stack(np.meshgrid(coords+point[0], coords+point[1]), 1)
    return linear_interpolation(image, grid)

def graph_training(model, image: RoadTracerImage, angle_samples, patch_size, step_size=10, merge_distance=5):
    graph = BaseNode()
    graph_layer = np.array([400, 400, 1], dtype=np.uint8)

    for starting_point in image.road_samples:
        if init_node.distance_to(starting_point) < merge_distance:
            continue
            
        root = Node(starting_point)
        init_node.append(root)

        stack = [root]

        while stack:
            top_node = stack[-1]

            model_input = np.stack((get_patch(image.image, top_node.p, patch_size), graph_layer/255.0), 2)
            angle_pred_dist, stop_dist, _ = model(utils.np_to_tensor(to_pytorch_img(model_input[None], "cuda")))
            angle_pred_dist = angle_pred_dist[0]
            stop_dist = stop_dist[0]

            # oracle output
            angle_true = get_oracle_prediction(image, starting_point, graph, merge_distance)

            if angle_true is not None:
                angle_loss = torch.functional.cross_entropy(angle_pred_dist, (angle_true/2/pi*angle_samples).astype(np.int32))
                stop_loss = torch.functional.binary_cross_entropy(stop_dist, 0.0)
            else:
                angle_loss = 0
                stop_loss = torch.functional.binary_cross_entropy(stop_dist, 1.0)
            
            total_loss = angle_loss + stop_loss
            yield total_loss

            if angle_true is None:
                stack.pop()
            else:
                node = Node(get_circle_sample_points(angle_samples, step_size)[np.argmax(angle_pred_dist.detach().cpu().numpy())])
                top_node.children.append(node)
                cv2.line(graph_layer, top_node.p, node.p, 255, 1.0)
                stack.append(node)




class TrainingLoop:
    def __init__(self, dataset, epochs, name="run"):
        self.dataset = dataset
        self.epochs = epochs
        self.name = name
        self.logdir = f'./tensorboard/{self.name}-{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
        self.writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    def train(self, model, optimizer):
        batches_done = 0
        # training loop
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            all_losses = []
            self.model.train()
            for loss, metrics in self._train_generator():
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.writer.add_scalar("loss", loss.item(), batches_done)
                self.writer.flush()
                batches_done += 1
            
            self.model.eval()
            val_metrics = {}
            for metrics in self._eval_generator():
                for m, mv in metrics.items():
                    if m in val_metrics:
                        val_metrics[m] = [mv]
                    else:
                        val_metrics[m].append(mv)

            for m, mv in val_metrics.items():
                self.writer.add_scalar(f"val_{m}", np.mean(mv), batches_done)
            self.writer.flush()

            # initialize metric list
            metrics = {'loss': [], 'val_loss': []}
            for k, _ in metric_fns.items():
                metrics[k] = []
                metrics['val_'+k] = []

            # training
            # validation
            # model.eval()
            # first = True
            # with torch.no_grad():  # do not keep track of gradients
            #     for (x, y) in eval_dataloader:
            #         y_hat = model(x)  # forward pass
            #         loss = loss_fn(y_hat, y)

            #         if first:
            #             writer.add_images("input", x, epoch)
            #             writer.add_images("output", y_hat[:, None, None], epoch)
            #             writer.add_images("ground truth", y[:, None, None], epoch)
            #             first = False

            #         # log partial metrics
            #         metrics['val_loss'].append(loss.item())
            #         for k, fn in metric_fns.items():
            #             metrics['val_'+k].append(fn(y_hat, y))

            # # summarize metrics, log to tensorboard and display
            # history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
            # for k, v in history[epoch].items():
            #     writer.add_scalar(k, v, epoch)
            
            # writer.flush()
            # print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
            # utils.show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

    def _train_generator(self):
        raise NotImplementedError()

    def _eval_generator(self):
        raise NotImplementedError()



class GraphTrainingLoop (TrainingLoop):
    def __init__(self, model, dataset, epochs, angle_samples, patch_size):
        super().__init__(model, dataset, epochs, "roadtracer_graph")
        self.angle_samples = angle_samples
        self.patch_size = patch_size
        self.step_size = step_size
        self.merge_size = merge_size

    def _train_generator(self):
        for image in tqdm.tqdm(self.dataset, ncols="150", desc="Images processed"):
            total_loss, batch_size = 0.0, 0
            for loss, metrics in graph_training(self.model, image, self.angle_samples, self.patch_size, self.step_size, self.merge_size):
                total_loss += loss
                batch_size += 1

                if batch_size == self.batch_size:
                    yield total_loss / batch_size, 
                    total_loss = 0.0
                    batch_size = 0
    
    def _eval_generator(self):
        return

