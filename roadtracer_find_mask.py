import argparse
from itertools import chain
from random import random, randint, shuffle

import cv2
import numpy as np
from matplotlib import pyplot as plt

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

parser.add_argument("--image_size", default=256, type=int, help="The size to which we resize data before training the model")

parser.add_argument("--population_size", default=1, type=int, help="The size of the population")
parser.add_argument("--epochs", default=10000, type=int, help="How many epochs to run the evolution for")




class SolutionPoint:
    def __init__(self, p):
        self.p = p

    def mutate(self, resolution):
        mutation = random()
        pos = np.random.randint(0, resolution, 2) if mutation < 0.1 else (self.p + np.random.randint(-10, 10, 2) if mutation < 0.2 else self.p.copy())
        return SolutionPoint(pos)

    def inside(self, resolution):
        return self.p[0] < resolution and self.p[1] < resolution and self.p[0] > 0 and self.p[1] > 0


class SolutionEdge:
    def __init__(self, src, dst, thick = None):
        self.src = src
        self.dst = dst
        self.thick = thick if thick is not None else SolutionEdge.random_thickness()

    def mutate(self):
        mutation = random()
        thick = SolutionEdge.random_thickness() if mutation < 0.1 else (max(1, self.thick+randint(-5, 5))) if mutation < 0.2 else self.thick 
        return SolutionEdge(self.src, self.dst, thick)
    
    @staticmethod
    def random_thickness():
        return randint(1, 30)


class Solution:
    def __init__(self, args, points, edges):
        self.points = points
        self.edges = edges
        self.args = args

    def render(self):
        x = np.zeros([self.args.image_size, self.args.image_size], dtype=np.uint8)
        point_thicknesses = [0 for _ in self.points]
        for e in self.edges:
            cv2.line(x, self.points[e.src].p, self.points[e.dst].p, 255, e.thick)
            point_thicknesses[e.src] = max(point_thicknesses[e.src], e.thick)
            point_thicknesses[e.dst] = max(point_thicknesses[e.dst], e.thick)
        for thick, p in zip(point_thicknesses, self.points):
            if thick > 0:
                cv2.circle(x, p.p, thick//2, 255)
                
        plt.imshow(x)
        plt.draw()
        plt.waitforbuttonpress()
                
        return x

    def fitness(self, ref_actions, ref_angles):
        sample_resolution = ref_actions.shape[0]
        
        render = self.render()

        #!! distance transform on original image? unsure how it should work now!
        dist = cv2.distanceTransform(render, 2, 3)

        sample_points = np.linspace(0, self.args.image_size, sample_resolution)
        points = np.stack(np.meshgrid(sample_points, sample_points), axis=2).reshape(-1, 2)
        samples = get_circle_sample_points(points, ref_angles.shape[-1], self.args.step_distance)
        scores = linear_interpolation(dist, samples).reshape(-1, sample_resolution, sample_resolution, ref_angles.shape[-1])

        norm_factor = np.max(scores, -1, keepdims=True)
        scores /= np.where(norm_factor > 0, norm_factor, 1.0)

        #... mse over all fields
        action_error = self.args.actions_K * np.mean(np.square(cv2.resize(render, (sample_resolution, sample_resolution), interpolation=cv2.INTER_LINEAR) / 255.0 - ref_actions))
        angle_error = self.args.angles_K * np.mean(np.square(scores - ref_angles))
        
        point_edge_count = [0 for _ in self.points]
        for e in self.edges:
            point_edge_count[e.src] += 1
            point_edge_count[e.dst] += 1
        
        
        regularization_error = 0.0
        point_count = 0
        if self.points:
            for p, edge_count in zip(self.points, point_edge_count):
                if edge_count == 1 and p.inside(self.args.image_size):
                    regularization_error += self.args.dead_end_regularization_K
                if edge_count == 2:
                    regularization_error += self.args.midpoint_regularization_K
                if edge_count > 0:
                    point_count += 1
            regularization_error /= len(self.points)

        point_error = point_count * self.args.point_regularization_K

        total_error = action_error + angle_error + regularization_error + point_error

        return total_error


    def mutate(self):
        new_points = [p.mutate(self.args.image_size) for p in self.points]
        new_edges = [e.mutate() for e in self.edges]
        
        if random() < 0.3: # add new point
            new_points.append(SolutionPoint(np.random.randint(0, self.args.image_size, 2)))
            
            if len(self.points) > 1:
                src, dst = len(new_points) - 1, randint(0, len(new_points) - 1)
                if src != dst:
                    new_edges.append(SolutionEdge(src, dst))
        
        if random() < 0.3 and self.edges: # remove edge
            new_edges.pop(randint(0, len(self.edges)-1))
        
        if random() < 0.3 and len(new_points) > 1: # add new edge
            src, dst = randint(0, len(self.points)-1), randint(0, len(self.points)-1)
            new_edges.append(SolutionEdge(src, dst))
        return Solution(self.args, new_points, new_edges)


    def crossover(self, other):
        new_points = []
        new_edges = []
        
        for p1, p2 in zip(self.points, other.points):
            new_points.append(p1 if random() > 0.5 else p2)
        
        new_points.extend(other.points[len(self.points):])
        new_points.extend(self.points[len(other.points):])
        
        for e in chain(self.edges, other.edges):
            if random() < 0.5:
                continue
            new_edges.append(e)
        
        return Solution(self.args, new_points, new_edges)



def run_evolution(population_size, epoch_count, actions, angles, args):
    population = [Solution(args, [], []) for _ in range(population_size)]
    
    best_fitness = 1e9
    best_solution = None
    
    for epoch in range(epoch_count):
        shuffle(population)
        new_population = [p.mutate() for p in population] + [p1.crossover(p2) for p1, p2 in zip(population[0::2], population[1::2])] + population
        fitnesses = [p.fitness(actions, angles) for p in new_population]
        order = np.argsort(fitnesses)[::-1]
        population = [new_population[i] for i in order[:population_size]]
        
        if fitnesses[order[0]] < best_fitness:
            best_fitness = fitnesses[order[0]]
            best_solution = population[0]
            plt.imshow(best_solution.render())
            plt.draw()
            plt.waitforbuttonpress()
        
        print (f"Epoch {epoch:< 3d}/{epoch_count:< 3d}: Best fitness found so far: {best_fitness}")
        


if __name__ == "__main__":
    args = parser.parse_args()
    actions = np.load(f"{args.image_fname}_actions.npy")
    angles = np.load(f"{args.image_fname}_angles.npy")
    run_evolution(args.population_size, args.epochs, actions, angles, args)
