from time import time
import numpy as np
import matplotlib.pyplot as plt

from mcc.utils import *
from mcc.parameters import *
from mcc.traversal import TraversalWithKeys
from mcc.graph import Graph, Node

if __name__ == '__main__':
    circles, obstacles = generate_circle(4, 2, 2)
    def get_points_in_a_circle(center, radius: float):
        """Returns points in a circle with some added gaussian noise"""
        thetas = np.linspace(0, 2 * np.pi, COUNT)
        states = []
        cx, cy = center

        for i, theta in enumerate(thetas):
            # np.random.seed(i)
            # Get the same random noise to get print information
            x = cx + radius * np.cos(theta) + np.random.randn() * NOISE
            y = cy + radius * np.sin(theta) + np.random.randn() * NOISE

            states.append(np.array([x, y]).reshape(-1, 1))

        return states

    graph_all = Graph()
    startTime = time()
    for circle in circles:
        center, radius = circle
        states = get_points_in_a_circle(center, radius)
        center = np.array(center).reshape(-1, 1)
        for state in states:
            graph_all.add_vertex(Node(state, center))

    graph_all.construct_kd_tree()
    
    traverser = TraversalWithKeys(graph=graph_all, centers=obstacles, SWITCH=True)
    traverser.make_graph(NN)

    
    start = graph_all.vertices[10]
    path = traverser.traverse(start)
    endTime = time() - startTime
    cost = traverser.calculate_path_cost(path)

    print(endTime, cost)
    
    # Plot the graph
    fig, ax = plt.subplots(figsize=(10, 8))

    X = [v.state[0][0] for v in graph_all.vertices]
    Y = [v.state[1][0] for v in graph_all.vertices]
    for i in range(len(path)):
        # graph_all.plot_graph(ax)\
        ax.scatter(X, Y, color="red")
        ax.set_title("Information Cost")
        ax.scatter(start.state[0][0], start.state[1][0], color="blue", s=50)
        graph_all.plot_vertices(ax, path[0:i+1])

        ax.scatter(path[i].state[0][0], path[i].state[1][0], marker="*", s=75, color="black")

        # plt.savefig(f"./images/{i}.png")
        plt.pause(0.001)
        plt.cla()