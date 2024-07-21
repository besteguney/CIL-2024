from math import pi
from collections.abc import Iterator

import numpy as np
import torch
import parameters as params
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import utils
from roadtracer_graph import BaseNode, Node, RoadTracerImage, get_oracle_prediction, filter_samples
from roadtracer_utils import get_circle_sample_points, get_patch, to_pytorch_img
from roadtracer_logging import Logger, LogMetrics


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



class GraphStepResult:
    def __init__(self, graph_layer, loss = None):
        self.graph_layer = graph_layer
        self.loss = loss


def generate_roadtracer_graph(image, starting_positions, model, metrics: "LogMetrics",
        angle_samples, patch_size, step_size, merge_distance, run_training, oracle_image=None, angle_train_single_target_only=False, use_oracle_for_step=False) -> Iterator[GraphStepResult]:
    graph = BaseNode()
    graph_layer = np.zeros([400, 400, 1], dtype=np.uint8)
    stepi = 0

    for starting_point in starting_positions:
        if graph.distance_to(starting_point) < merge_distance:
            continue
            
        root = Node(starting_point)
        graph.children.append(root)

        stack = [root]

        while stack:
            top_node = stack[-1]

            

            if not use_oracle_for_step:
                model_input = np.concatenate((get_patch(image, top_node.p, patch_size), get_patch(graph_layer/255.0, top_node.p, patch_size)), 2)
                action_dist, angle_dist, _ = model(utils.np_to_tensor(to_pytorch_img(model_input)[None], "cuda").float())
                action_dist = action_dist[0]
                angle_dist = angle_dist[0]
            else:
                # oracle output
                angle_scores = get_oracle_prediction(oracle_image, top_node.p, graph, angle_samples, step_size, merge_distance, false)
                angle_scores, idx = filter_samples(angle_scores, 4, np.arange(angle_samples))
                action_dist = torch.tensor([1.0, 0.0]) if angle_target is not None else torch.tensor([0.0, 1.0])
                angle_dist = torch.zeros(angle_samples)
                angle_dist[idx] = 1.0

            teacher_force_end = False

            if run_training:
                # oracle output
                angle_target = get_oracle_prediction(oracle_image, top_node.p, graph, angle_samples, step_size, merge_distance, angle_train_single_target_only)
                # We don't want to end! 
                if angle_target is not None:
                    # angle_label = int(angle_true/2/pi*angle_samples)
                    # angle_target = torch.zeros([angle_samples])
                    # angle_target[angle_label] = 1.0
                    angle_loss = torch.nn.functional.mse_loss(angle_dist, torch.from_numpy(angle_target).float().cuda())
                    action_loss = torch.nn.functional.cross_entropy(action_dist, torch.FloatTensor([1.0, 0.0]).cuda())
                else: # we want to end
                    angle_loss = 0
                    action_loss = torch.nn.functional.cross_entropy(action_dist,  torch.FloatTensor([0.0, 1.0]).cuda())
                    teacher_force_end = True
                

                total_loss = 50 * angle_loss + action_loss
                yield GraphStepResult(graph_layer, total_loss)
            else:
                yield GraphStepResult(graph_layer)

            # either there is nothing to be generated according to the oracle, or the model wants to stop
            if teacher_force_end or action_dist[0] < action_dist[1] or len(stack) > 500:
                stack.pop()
                metrics.log_metric("step_percentage", 0.0)
            else:
                metrics.log_metric("step_percentage", 1.0)
                points_sorted = get_circle_sample_points(top_node.p, angle_samples, step_size)[np.argsort(angle_dist.detach().cpu().numpy())[::-1]]
                for p in points_sorted:
                    if graph.distance_to(p) > merge_distance and np.all(p >= 0) and np.all(p <= 400): 
                        node = Node(p)
                        top_node.children.append(node)
                        draw_line(graph_layer, top_node.p, node.p, 255, 1)
                        stack.append(node)
                        break
                else:
                    stack.pop()

    # metrics.log_image("base_image", np.moveaxis(image.image, -1, 0))
    # metrics.log_image("search_image", image.search[None])
    # metrics.log_image("rendered_graph", np.moveaxis(graph_layer, -1, 0))
    # vertices_render = np.zeros([400, 400, 1], np.uint8)
    # for v in graph.get_vertices():
    #     cv2.circle(vertices_render, (int(v[0]), int(v[1])), 3, 255, -1)
    # metrics.log_image("rendered_vertices", np.moveaxis(vertices_render, -1, 0))

    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(graph_layer)
    # ax2.imshow(image.image)
    # ax2.scatter(*zip(*graph.get_vertices()))
    # plt.draw()
    # plt.waitforbuttonpress()
    # plt.close()


class TrainingLoop:
    def __init__(self, logger, train_images: list[RoadTracerImage], eval_images: list[RoadTracerImage], epochs: int, save_freq: int, name: str="run"):
        self.train_images = train_images
        self.eval_images = eval_images
        self.epochs = epochs
        self.save_freq = save_freq
        self.name = name
        self.logger = logger

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
                first = False
            
            model.eval()
            val_metrics = LogMetrics()
            for metrics in self._eval_generator(model, "Validation"):
                val_metrics.log_metrics(metrics)
            
            self.logger.log_metrics(val_metrics, batches_done, "val_")
            
            if (epoch+1) % self.save_freq == 0:
                self.logger.save_model(model, epoch)

    def _train_generator(self, model, step_description: str) -> Iterator[torch.Tensor, LogMetrics]:
        raise NotImplementedError()

    def _eval_generator(self, model, step_description: str) -> Iterator[torch.Tensor, LogMetrics]:
        raise NotImplementedError()



class GraphTrainingLoop (TrainingLoop):
    def __init__(self, logger, train_images, val_images, batch_size, epochs, angle_samples, patch_size, step_size, merge_distance, single_angle_target):
        super().__init__(logger, train_images, val_images, epochs, 1, "roadtracer_graph")
        self.batch_size = batch_size
        self.generate_graph_kwargs = {
            "angle_samples": angle_samples,
            "patch_size": patch_size,
            "step_size": step_size,
            "merge_distance": merge_distance,
            "angle_train_single_target_only": single_angle_target
        }
        

    def _train_generator(self, model, step_description: str):
        metrics = LogMetrics()
        for image in tqdm(self.train_images, ncols=150, desc=step_description):
            total_loss, batch_size = 0.0, 0
            last_result = None
            for step_result in generate_roadtracer_graph(
                    image.image, image.road_samples, model, metrics,
                    **self.generate_graph_kwargs,
                    oracle_image=image.distance, run_training=True):
                
                total_loss += step_result.loss
                batch_size += 1

                if batch_size == self.batch_size:
                    yield total_loss / batch_size, metrics
                    metrics.reset()
                    total_loss = 0.0
                    batch_size = 0
                
                last_result = step_result
                
            metrics.log_image("graph_result", step_result.graph_layer)
            metrics.log_image("graph_target", image.search)
            metrics.log_image("image", image.image)
    
    def _eval_generator(self, model, step_description: str):
        for image in tqdm(self.eval_images, ncols=150, desc=step_description):
            metrics = LogMetrics()
            # last_result = None
            # for step_result in generate_roadtracer_graph(
            #         image.image, image.road_samples, model, metrics,
            #         **self.generate_graph_kwargs,
            #         run_training=False):
            #     last_result = step_result
            # metrics.log_image("graph_result", last_result.graph_layer)
            # metrics.log_image("graph_target", image.search)
            # metrics.log_image("image", image.image)
            yield metrics

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


