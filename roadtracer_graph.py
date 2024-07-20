import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
import cv2
from math import sin, cos, pi, sqrt, inf
from matplotlib.collections import LineCollection
from scipy.ndimage import maximum_filter



DIST_SAMPLES = 1
ANGLE_SAMPLES = 60
STEP_SIZE = 10.0
DIST_FROM_PERCENT = 1.0
DIST_TO_PERCENT = 1.0

DIST_SAMPLES_REQUIRED = 1
VALID_POINT_MAX_RANGE = 4
MERGE_DIST = STEP_SIZE / 1.3


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

# return a [distances, angles, 2] array of points describing sampling locations on a circle
def circle_sample_points(angle_samples, dist_samples, sample_radius):        
    angles = np.linspace(0, 2*pi, angle_samples, endpoint=False, dtype=np.float32)
    directions = np.stack((np.cos(angles), np.sin(angles)), 1)  #shape [angles, 2]
    if dist_samples is None:
        return np.array([[sample_radius]]) * directions

    distances = np.linspace(sample_radius, 0, dist_samples, endpoint=False, dtype=np.float32)
    return directions[np.newaxis] * distances[:, np.newaxis, np.newaxis]


def graph_step(search_image, point, angle_samples, distance_samples, sample_radius):
    w, h = search_image.shape[:2]

    samples = circle_sample_points(angle_samples, distance_samples, sample_radius) + point.astype(np.float32)
    scores = linear_interpolation(search_image, samples) # [distances, angles]
    all_scores = np.sum(scores, 0)


    # all_scores = []
    # for angle in angles:
    #     score, samples = 0, 0
    #     s, c = sin(angle), cos(angle)
    #     for dist in np.linspace(STEP_SIZE * DIST_FROM_PERCENT, STEP_SIZE * DIST_TO_PERCENT, DIST_SAMPLES):
    #         x = px + dist * c
    #         y = py + dist * s
    #         if x < 0 or y < 0 or x >= w or y >= h:
    #             continue
    #         score += linear_interpolation(search_image, np.array([x, y]))
    #         samples += 1
    #     final_score = 0.0 if samples < DIST_SAMPLES_REQUIRED else score / samples
    #     all_scores.append(final_score)
    
    return all_scores, circle_sample_points(angle_samples, None, sample_radius)


def filter_samples(next_points, scores, filter_range, score_threshold=0.4):
    angle_samples = len(scores)
    valid_samples = []
    for i in range(ANGLE_SAMPLES):
        score = scores[i]
        if score >= max(scores[j % angle_samples] for j in range(i - filter_range, i + filter_range + 1)) and score > score_threshold:
            valid_samples.append(next_points[i])

    return valid_samples

        
def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)



class BaseNode:
    def __init__(self):
        self.children = []
        self.depth = None

    def get_vertices(self):
        return sum([c.get_vertices() for c in self.children], start=[])

    def get_edges(self):
        return sum([c.get_edges() for c in self.children], start=[])

    def _compute_depth(self):
        for c in self.children:
            c._compute_depth()
        self.depth = max([c.depth for c in self.children] + [0]) + 1
    
    def prune(self):
        if self.depth is None:
            self._compute_depth()
        has_main_branch = any(c.depth > 2 for c in self.children)
        if has_main_branch:
            self.children = [c for c in self.children if c.depth > 2]
        for c in self.children:
            c.prune()

    def distance_to(self, p):
        return min([c.distance_to(p) for c in self.children] + [inf])

    def size(self):
        return 1 + sum(c.size() for c in self.children)


class Node (BaseNode):
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def get_vertices(self):
        return super().get_vertices() + [[self.p[1], self.p[0]]]

    def get_edges(self):
        return super().get_edges() + [[(c.p[1], c.p[0]), (self.p[1], self.p[0])] for c in self.children]

    def distance_to(self, p):
        return min(super().distance_to(p), distance(p, self.p))



def plot_graph(graph_root, search_image, active_node = None):
    lc = LineCollection(graph_root.get_edges(), color="red")
    fig, ax = plt.subplots()
    ax.add_collection(lc)

    vertices = graph_root.get_vertices()
    if len(vertices):
        vx, vy = zip(*vertices)
        plt.scatter(vx, vy, color="yellow")
    if active_node is not None:
        plt.scatter([active_node.p[1]], [active_node.p[0]], color="red")
    plt.imshow(search_image)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close()



def get_search_image(image):
    dist_image = cv2.distanceTransform(image, 2, 3)
    search_image = maximum_filter(dist_image, size=(7, 7), mode="constant") - dist_image < 2.5
    search_image = np.where(np.logical_and(dist_image > 2, search_image), 1.0, 0.0)
    return dist_image, search_image






def find_graph(img, angle_samples, distance_samples, sample_radius):
    step_size = 10

    dist_image, search_image = get_search_image(img)

    graph_root = BaseNode()

    start_xs, start_ys = np.nonzero(search_image)
    order = np.argsort(dist_image[start_xs, start_ys])[::-1]

    for start_point in zip(start_xs[order], start_ys[order]):
        if graph_root.distance_to(start_point) < step_size:
            continue

        root = Node(start_point)
        graph_root.children.append(root)
        

        queue = [root]
        while queue:
            v = queue[0]
            queue = queue[1:]
            next_scores, next_points = graph_step(search_image, v.p, angle_samples, distance_samples, sample_radius)
            next_points = filter_samples(next_points, next_scores)
            for next_p in next_points:
                if graph_root.distance_to(next_p) > MERGE_DIST:
                    n = Node(next_p)
                    v.children.append(n)
                    queue.append(n)
        if root.size() < 4:
            graph_root.children.pop()
            
    plot_graph(graph_root, search_image)



if __name__ == "__main__":
    for img_name in glob.glob("ethz-cil-road-segmentation-2024/training/groundtruth/*"):
        img = np.asarray(Image.open(img_name))
        find_graph(img)
        


