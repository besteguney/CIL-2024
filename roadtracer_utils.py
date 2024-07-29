import numpy as np
import cv2
import scipy.ndimage
from math import pi, sin, cos
import torch

def from_pytorch_img(img):
    return np.moveaxis(img.numpy(), -3, -1)

def to_pytorch_img(img):
    if len(img.shape) != 2:
        img = np.moveaxis(img, -1, -3)
    return torch.from_numpy(img).float()

def get_search_image(image):
    dist_image = cv2.distanceTransform(image, 2, 3)
    search_image = scipy.ndimage.maximum_filter(dist_image, size=(7, 7), mode="constant") - dist_image < 2.5
    search_image = np.where(np.logical_and(dist_image > 2, search_image), 1.0, 0.0)
    return dist_image, search_image


class RoadTracerImage:
    def __init__(self, img, target, distance=None, search=None):
        self.image = img
        if distance is None or search is None:
            self.distance, self.search = get_search_image((255*target).astype(np.uint8))
        else:
            self.distance, self.search = distance, search
        self.target = target

        self.road_samples = np.stack(np.nonzero(self.search != 0), 1)
        self.road_samples = self.road_samples[np.argsort(self.distance[self.road_samples[..., 0], self.road_samples[..., 1]])]
        self.negative_samples = np.stack(np.nonzero(self.search == 0), 1)
    

def linear_interpolation(img, points):
    px, py = points[..., 0], points[..., 1]
    w, h = img.shape[:2]
    outside = np.logical_or(np.logical_or(px < 0, px >= w), np.logical_or(py < 0, py >= h))
    
    x0 = np.minimum(np.maximum(0, px.astype(np.int32)), w-1)
    y0 = np.minimum(np.maximum(0, py.astype(np.int32)), h-1)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    a, b = px - x0, py - y0
    if len(img.shape) == 3: # 3 dim image; [w, h, c]
        a = a[..., np.newaxis]
        b = b[..., np.newaxis]
        outside = outside[..., np.newaxis]
    result = img[x0, y0] * (1-a) * (1-b) + img[x1, y0] * a * (1-b) + img[x0, y1] * (1-a) * b + img[x1, y1] * a * b
    return np.where(outside, 0.0, result)


def angle_sample_points(angle_samples):
    return np.linspace(0, 2*pi, angle_samples, endpoint=False, dtype=np.float32)


# return a [distances, angles, 2] array of points describing sampling locations on a circle
def get_circle_sample_points(center, angle_samples, sample_radius, dist_samples = None):        
    angles = angle_sample_points(angle_samples)
    directions = np.stack((np.cos(angles), np.sin(angles)), 1)  #shape [angles, 2]
    if dist_samples is None:
        if center.ndim == 2:
            center = center[:, None]
        return center + np.array([[sample_radius]]) * directions

    distances = np.linspace(sample_radius, 0, dist_samples, endpoint=False, dtype=np.float32)
    return center + directions[np.newaxis] * distances[:, np.newaxis, np.newaxis]


def get_patch(image, point, patch_size, rotation=None):
    step = (patch_size-1)/2
    coords = np.linspace(-step, step, patch_size)
    grid = np.stack(np.meshgrid(coords, coords), 2)   # size [patch_size, patch_size, 2]
    if rotation is not None and rotation != 0:
        grid = grid @ np.array([[cos(rotation), -sin(rotation)], [sin(rotation), cos(rotation)]])
    if point.ndim == 1:
        grid = grid + point
    else:
        grid = np.moveaxis(grid[..., None, :] + point, -2, 0)
    return linear_interpolation(image, grid)
