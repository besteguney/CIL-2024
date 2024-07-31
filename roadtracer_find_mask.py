import argparse

import cv2
import numpy as np

from roadtracer_utils import linear_interpolation, get_circle_sample_points 


parser = argparse.ArgumentParser()
parser.add_argument("--file_path", default="ethz-cil-road-segmentation-2024", type=str, help="Path to images to run inference for")

parser.add_argument("--step_distance", default=16.0, type=float, help="The length of the edges in the generated graph")

parser.add_argument("--image_fname", default="roadtracer_results/satimage_0", type=str, help="The angle image to find roads in")

parser.add_argument("--actions_K", default=10.0, type=float, help="Loss term, actions coefficient")
parser.add_argument("--angles_K", default=10.0, type=float, help="Loss term, angles coefficient")
parser.add_argument("--wrong_count_K", default=5.0, type=float, help="Loss term, actions coefficient")
parser.add_argument("--midpoint_regularization_K", default=5.0, type=float, help="Loss term, actions coefficient")
parser.add_argument("--dead_end_regularization_K", default=5.0, type=float, help="Loss term, actions coefficient")

parser.add_argument("--point_regularization_K", default=0.1, type=float, help="Loss term, point regularization")




MIDPOINT, JUNCTION, DEAD_END, EDGE_POINT = 0, 1, 2

class SolutionPoint:
    def __init__(self, type_, p):
        self.type = type_
        self.p = p


class SolutionEdge:
    def __init__(self, src, dst, thick):
        self.src = src
        self.dst = dst
        self.thick = thick


class Solution:
    def __init__(self, args, points, edges):
        self.points = points
        self.edges = edges
        self.resolution = resolution
        self.args = args

    def render(self):
        x = np.zeros([self.resolution, self.resolution], dtype=np.uint8)
        point_thicknesses = [0.0 for p in self.points]
        for e in self.edges:
            cv2.line(x, self.points[e.src].p, self.points[e.dst].p, 255, e.thick)
            point_thicknesses[e.src] = max(point_thicknesses[e.src], e.thick)
            point_thicknesses[e.dst] = max(point_thicknesses[e.dst], e.thick)
        for thick, p in zip(point_thicknesses, self.points):
            cv2.circle(x, p.p, thick/2, 255)
        return x

    def fitness(self, ref_actions, ref_angles):
        render = self.render()

        dist = cv2.distanceTransform(render, 2, 3)

        samples = get_circle_sample_points(point, ref_angles.shape[-1], self.args.step_distance)
        scores = linear_interpolation(search_image, samples)

        scores /= np.max(scores, -1, keepdims=True)

        #... mse over all fields
        action_error = self.args.actions_K * np.mean(np.square(render / 255.0 - ref_actions))
        angle_error = self.args.angles_K * np.mean(np.square(scores - ref_angles))
        
        point_edge_count = [0 for _ in self.points]
        for e in self.edges:
            point_edge_count[e.src] += 1
            point_edge_count[e.dst] += 1
        
        wrong_count_error = 0.0
        regularization_error = 0.0
        for p, edge_count in zip(self.points, point_edge_count):
            err = 0
            if p.type == EDGE_POINT:
                err = abs(edge_count-1) # one road exactly
            if p.type == MIDPOINT:
                err = abs(edge_count - 2) # 2 roads exactly
                regularization_error += self.args.midpoint_regularization_K
            if p.type == JUNCTION:
                err = max(3-edge_count, 0) # 3 or more roads
            if p.type == DEAD_END:
                err = abs(edge_count-1) # one road exactly
                regularization_error += self.args.dead_end_regularization_K
            wrong_count_error += self.args.wrong_count_K * err
        wrong_count_error /= len(self.points)
        regularization_error /= len(self.points)

        point_error = len(self.points) * self.args.point_regularization_K

        total_error = action_error + angle_error + wrong_count_error + regularization_error + point_error

        return total_error
        




if __name__ == "__main__":
    args = parser.parse_args()
    actions = np.load(f"{args.image_fname}_actions.npy")
    angles = np.load(f"{args.image_fname}_angles.npy")
