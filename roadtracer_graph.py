import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
import cv2
from math import sin, cos, pi, sqrt, inf
from matplotlib.collections import LineCollection
from roadtracer_utils import RoadTracerImage, get_circle_sample_points, angle_sample_points, linear_interpolation


# DIST_SAMPLES = 1
# ANGLE_SAMPLES = 60
# STEP_SIZE = 10.0
# DIST_FROM_PERCENT = 1.0
# DIST_TO_PERCENT = 1.0

# DIST_SAMPLES_REQUIRED = 1
# VALID_POINT_MAX_RANGE = 4
# MERGE_DIST = STEP_SIZE / 1.3


def graph_step(search_image, point, angle_samples, sample_radius):
    w, h = search_image.shape[:2]

    samples = get_circle_sample_points(point, angle_samples, sample_radius)
    scores = linear_interpolation(search_image, samples) # [distances, angles]
    # all_scores = scores #np.sum(scores, 0)


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
    
    return scores, angle_sample_points(angle_samples), samples



def filter_samples(scores, filter_range, *more_data, score_threshold=1):
    angle_samples = len(scores)
    idx = []
    for i in range(angle_samples):
        score = scores[i]
        if score >= max(scores[j % angle_samples] for j in range(i - filter_range, i + filter_range + 1) if j != i) and score >= score_threshold:
            idx.append(i)

    return scores[idx], *[data[idx] for data in more_data]



def distance(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)



class BaseNode:
    def __init__(self):
        self.children = []
        self._depth = None

    def get_vertices(self):
        return sum([c.get_vertices() for c in self.children], start=[])

    def get_edges(self):
        return sum([c.get_edges() for c in self.children], start=[])

    def _compute_depth(self):
        for c in self.children:
            c._compute_depth()
        self._depth = max([c._depth for c in self.children] + [0]) + 1

    def depth(self):
        self._compute_depth()
        return self._depth
    
    def prune(self):
        if self._depth is None:
            self._compute_depth()
        has_main_branch = any(c._depth > 2 for c in self.children)
        if has_main_branch:
            self.children = [c for c in self.children if c._depth > 2]
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



def get_oracle_prediction(oracle_image, point, graph: BaseNode, angle_samples, step_distance, merge_distance, single_target_only=False):
    scores, angles, pts = graph_step(oracle_image, point, angle_samples, step_distance)
    if single_target_only: # only return one angle, as a numeric value
        scores, pts, idx = filter_samples(scores, 4, pts, np.arange(angle_samples))
        target = np.zeros(angle_samples)
        for i, p in zip(idx, pts):
            if graph.distance_to(p) > merge_distance:
                # the oracle target for the network is this point
                target[i] = 1.0
                return target
        return None
    else:
        scores = np.where(np.array([graph.distance_to(p) for p in pts]) > merge_distance, scores, 0.0)
        if np.max(scores) > 1.0:
            return scores / np.max(scores)
        return None









if __name__ == "__main__":
    for img_name in glob.glob("ethz-cil-road-segmentation-2024/training/groundtruth/*"):
        ground_truth = np.asarray(Image.open(img_name))
        point = np.random.uniform(0, 400, (2,))
        image = RoadTracerImage(None, ground_truth)

        # find_graph(image.search.astype(np.uint8), 60, 1, 10)

        scores, angles, pts = graph_step(image.distance, point, 64, 1, 10)
        scores, angles, pts = filter_samples(scores, 4, angles, pts)
        
        plt.imshow(image.distance)
        plt.scatter([point[1]], [point[0]], color="r")
        plt.scatter(pts[:, 1], pts[:, 0], c=scores, cmap="turbo")
        plt.show()
        
        # find_graph(img)
        


