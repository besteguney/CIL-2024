import datetime
from math import pi
from collections.abc import Iterator

import numpy as np
import torch
import parameters as params
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import utils
from roadtracer_graph import BaseNode, Node, RoadTracerImage, get_oracle_prediction
from roadtracer_utils import get_circle_sample_points, get_patch, to_pytorch_img



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





def draw_line(img, p1, p2, color, thickness: int):
    cv2.line(img, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), color, thickness)


def graph_training(model, metrics: "LogMetrics", image: RoadTracerImage, angle_samples, patch_size, step_size, merge_distance):
    graph = BaseNode()
    graph_layer = np.zeros([400, 400, 1], dtype=np.uint8)

    for starting_point in image.road_samples:
        if graph.distance_to(starting_point) < merge_distance:
            continue
            
        root = Node(starting_point)
        graph.children.append(root)

        stack = [root]

        while stack:
            top_node = stack[-1]

            model_input = np.concatenate((get_patch(image.image, top_node.p, patch_size), get_patch(graph_layer/255.0, top_node.p, patch_size)), 2)
            action_dist, angle_dist, _ = model(utils.np_to_tensor(to_pytorch_img(model_input)[None], "cuda").float())
            action_dist = action_dist[0]
            angle_dist = angle_dist[0]

            # oracle output
            angle_true = get_oracle_prediction(image, starting_point, graph, angle_samples, step_size, merge_distance)


            # We don't want to end! 
            if angle_true is not None:
                angle_label = int(angle_true/2/pi*angle_samples)
                angle_target = torch.zeros([angle_samples])
                angle_target[angle_label] = 1.0
                angle_loss = torch.nn.functional.mse_loss(angle_dist, angle_target.cuda())
                action_loss = torch.nn.functional.cross_entropy(action_dist, torch.FloatTensor([1.0, 0.0]).cuda())
                metrics.log_metric("step_percentage", 1.0)
            else: # we want to end
                angle_loss = 0
                action_loss = torch.nn.functional.cross_entropy(action_dist,  torch.FloatTensor([0.0, 1.0]).cuda())
                metrics.log_metric("stop_percentage", 0.0)
            
            total_loss = 50 * angle_loss + action_loss
            yield total_loss

            # either there is nothing to be generated according to the oracle, or the model wants to stop
            if angle_true is None or action_dist[0] < action_dist[1]:
                stack.pop()
            else:
                points_sorted = get_circle_sample_points(top_node.p, angle_samples, step_size)[np.argsort(angle_dist.detach().cpu().numpy())[::-1]]
                for p in points_sorted:
                    if graph.distance_to(p) > merge_distance: 
                        node = Node(p)
                        top_node.children.append(node)
                        draw_line(graph_layer, top_node.p, node.p, 255, 1)
                        stack.append(node)
                        break
                else:
                    continue
                stack.pop()
    metrics.log_image("ground_truth", np.moveaxis(image.image, -1, 0))
    metrics.log_image("rendered_graph", np.moveaxis(graph_layer, -1, 0))
    vertices_render = np.zeros([400, 400, 1], np.uint8)
    for v in graph.get_vertices():
        cv2.circle(vertices_render, (int(v[1]), int(v[0])), 3, 255, -1)
    metrics.log_image("rendered_vertices", np.moveaxis(vertices_render, -1, 0))

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(graph_layer)
    # ax2.imshow(image.image)
    # ax2.scatter(*zip(*graph.get_vertices()))
    # plt.draw()
    # plt.waitforbuttonpress()
    # plt.close()





class MetricDict:
    def __init__(self):
        self.metrics = {}

    def add(self, name, val):
        if name in self.metrics:
            self.metrics[name].append(val)
        else:
            self.metrics[name] = [val]

    def get_avg(self):
        for name, vals in self.metrics.items():
            yield name, np.mean(vals)

    def get_stacked(self, axis=0):
        for name, vals in self.metrics.items():
            yield name, np.stack(vals, axis)

    def get_first(self, axis=0):
        for name, vals in self.metrics.items():
            yield name, vals[0]


    def get_list(self):
        for name, vals in self.metrics.items():
            yield name, vals

    def extend(self, name, vals):
        if name in self.metrics:
            self.metrics[name].extend(vals)
        else:
            self.metrics[name] = vals

    def reset(self):
        self.metrics = {}


class LogMetrics:
    def __init__(self):
        self.metrics = MetricDict()
        self.images = MetricDict()
        
    def log_metric(self, name, value):
        self.metrics.add(name, value)
    
    def log_image(self, name, img):
        self.images.add(name, img)

    def log_metrics(self, metrics: "LogMetrics"):
        for name, vals in metrics.metrics.get_list():
            self.metrics.extend(name, vals)
        for name, imgs in metrics.images.get_list():
            self.images.extend(name, imgs)
    
    def reset(self):
        self.metrics.reset()
        self.images.reset() 




class Logger:
    def __init__(self, name):
        self.log_dir = f'./log/{name}-{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
        self.writer = SummaryWriter(self.log_dir)

    def log_metrics(self, metrics: LogMetrics, step: int, prefix: str=""):
        for name, val in metrics.metrics.get_avg():
            self.writer.add_scalar(prefix + name, val, step)

        for name, val in metrics.images.get_first():
            self.writer.add_image(prefix + name, val, step)

        self.writer.flush()

    def save_model(self, model, epoch):
        torch.save(model.state_dict(), f"{self.log_dir}/model_{epoch:0<3d}.pt")





class TrainingLoop:
    def __init__(self, train_images: list[RoadTracerImage], eval_images: list[RoadTracerImage], epochs: int, save_freq: int, name: str="run"):
        self.train_images = train_images
        self.eval_images = eval_images
        self.epochs = epochs
        self.save_freq = save_freq
        self.name = name
        self.logger = Logger(name)  # tensorboard writer (can also log images)

    def train(self, model, optimizer):
        batches_done = 0
        # training loop
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            model.train()
            for loss, metrics in self._train_generator(model, f"Epoch {epoch+1}/{self.epochs}"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics.log_metric("loss", loss.item())
                self.logger.log_metrics(metrics, batches_done)
                batches_done += 1
            
            model.eval()
            val_metrics = LogMetrics()
            for metrics in self._eval_generator(model, "Validation"):
                val_metrics.add_metrics(metrics)
            
            self.writer.log_metrics(val_metrics, batches_done, "val_")
            
            if (epoch+1) % save_freq == 0:
                self.logger.save_model(self.model, epoch)

    def _train_generator(self, model, step_description: str) -> Iterator[torch.Tensor, LogMetrics]:
        raise NotImplementedError()

    def _eval_generator(self, model, step_description: str) -> Iterator[torch.Tensor, LogMetrics]:
        raise NotImplementedError()



class GraphTrainingLoop (TrainingLoop):
    def __init__(self, train_images, val_images, batch_size, epochs, angle_samples, patch_size, step_size, merge_size):
        super().__init__(train_images, val_images, epochs, 1, "roadtracer_graph")
        self.batch_size = batch_size
        self.angle_samples = angle_samples
        self.patch_size = patch_size
        self.step_size = step_size
        self.merge_size = merge_size
        

    def _train_generator(self, model, step_description: str):
        metrics = LogMetrics()
        for image in tqdm(self.train_images, ncols=150, desc=step_description):
            total_loss, batch_size = 0.0, 0
            for loss in graph_training(model, metrics, image, self.angle_samples, self.patch_size, self.step_size, self.merge_size):
                total_loss += loss
                batch_size += 1

                if batch_size == self.batch_size:
                    yield total_loss / batch_size, metrics
                    metrics.reset()
                    total_loss = 0.0
                    batch_size = 0
    
    def _eval_generator(self, model, step_description: str):
        pass
        # TODO early stopping + save best model
        # for image in tqdm.tqdm(self.eval_images, ncols=150, desc=step_description):
        #     continue





class DistanceTrainingLoop (TrainingLoop):
    def __init__(self, model, dataset, epochs, patch_size):
        super().__init__(dataset, epochs, "roadtracer_distance")
        self.loss = torch.nn.MSELoss()

    def _train_generator(self, step_description: str):
        for images, dist in tqdm(self.train_images, ncols=150, desc=step_description):
            pred_dist = self.model(images)
            loss = self.loss(pred_dist, dist)
            yield loss, {}
            
    def _eval_generator(self, step_description: str):
        for images, dist in tqdm(self.eval_images, ncols=150, desc=step_description):
            pred_dist = self.model(images)
            loss = self.loss(pred_dist, dist)
            yield {"loss": loss}


