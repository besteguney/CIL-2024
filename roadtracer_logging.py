import datetime

import wandb
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


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
    def __init__(self, name, use_wandb=True, wandb_entity=None, config={}):
        self.log_dir = f'./log/{name}-{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
        self.writer = SummaryWriter(self.log_dir)
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.require("core")
            wandb.init(
                project=name,
                entity=wandb_entity,
                config=config,
            )


    def log_metrics(self, metrics: LogMetrics, step: int, prefix: str=""):
        wandb_log = {}
        for name, val in metrics.metrics.get_avg():
            self.writer.add_scalar(prefix + name, val, step)
            if self.use_wandb:
                wandb_log[prefix + name] = val

        for name, val in metrics.images.get_first():
            self.writer.add_image(prefix + name, val, step, dataformats="HWC" if len(val.shape)==3 else "HW")
            if self.use_wandb:
                wandb_log[prefix + name] = wandb.Image(val)

        self.writer.flush()
        if self.use_wandb:
            wandb.log(wandb_log, step=step)


    def save_model(self, model, epoch):
        model_fname = f"{self.log_dir}/model_{epoch:0>3d}.pt"
        torch.save(model.state_dict(), model_fname)
        if self.use_wandb:
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(model_fname)
            wandb.log_artifact(artifact)