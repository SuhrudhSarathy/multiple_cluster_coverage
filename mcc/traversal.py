"""File that has algorithms for graph traversal"""
import numpy as np
from mcc.graph import Graph, Node
from mcc.parameters import *
from mcc.utils import *

class TraversalWithKeys:
    """Class that uses simple distance as traversal cost"""
    class Key:
        def __init__(self, k1, k2, k3):
            self.k1 = k1
            self.k2 = k2
            self.k3 = k3

        def __lt__(self, __ob):
            # preference 1: traversal cost
            if self.k1 == __ob.k1:
                # preference 2: Switching or not
                if self.k2 == __ob.k2:
                    # If the utility of those paths is also same, take the shortest path
                    return self.k3 < __ob.k3
                else:
                    return self.k2 < __ob.k2

            else:
                return self.k1 < __ob.k1

    def __init__(self, graph: Graph, centers, **kwargs):
        self.graph = graph
        self.centers = centers

        self.ALPHA = kwargs.get('ALPHA', 0.5)
        self.BETA = kwargs.get('BETA', 0.5)
        self.GAMMA = kwargs.get('GAMMA', 1)
        self.CLUSTER_DIST_MIN = kwargs.get('CLUSTER_DIST', 0.1)
        self.CLUSTER_DIST_MAX = kwargs.get('CLUSTER_DIST', 12)
        self.TRAVERSAL_PENALTY = kwargs.get('TRAVERSAL_PENALTY', 500)
        self.DIST_THRESHOLD = kwargs.get('DIST_THRESHOLD',2)

        self.SWITCH = kwargs.get('SWITCH', False)

        self.limit = self.graph.vertex_count * 10

        self.switching_locations = []
        self.switching_points = []

    def update_switching_points(self):
        """
        * Adds the switching connections to the graph
        * Given a bunch of centers, find the kruskall graph and add swithcing points
        between them
        """
        center_graph = Graph()
        for center in self.centers:
            c, r = center
            n = Node(np.array(c).reshape(-1, 1))
            n.radius = r
            center_graph.add_vertex(n)

        center_graph.construct_kd_tree()
        
        # Constructing the graph of centers
        for i, vertex in enumerate(center_graph.vertices):
            for j, vertex2 in enumerate(center_graph.vertices[i+1:]):
                center_graph.add_edge(vertex, vertex2, Node.distance(vertex.state, vertex2.state))
        
        # Obtain the kruskal graph
        kruskal_graph = center_graph.return_kruskal_graph()

        # now we got the graph with the centers
        for edge in kruskal_graph.edges:
            c1, c2 = edge
            # find the corresponding centers
            nearest_to_c1 = center_graph.get_k_nearest_neighbours(c1, 1)[0]

            nearest_to_c2 = center_graph.get_k_nearest_neighbours(c2, 1)[0]

            c1, r1 = nearest_to_c1.state, nearest_to_c1.radius
            c2, r2 = nearest_to_c2.state, nearest_to_c2.radius

            unit_vector = (c2-c1)/np.linalg.norm(c2-c1)

            point_near_c = c1 + r1 * unit_vector
            point_near_co = c2 - r2 * unit_vector


            n1 = self.graph.get_k_nearest_neighbours(point_near_c.reshape(1, -1), 1)[0]
            n2 = self.graph.get_k_nearest_neighbours(point_near_co.reshape(1, -1), 1)[0]

            n1.isSwitching = True
            n2.isSwitching = True
            self.graph.add_edge(n1, n2)
            self.switching_locations.append([n1, n2])

            if n1 not in self.switching_points:
                self.switching_points.append(n1)
            if n2 not in self.switching_points:
                self.switching_points.append(n2)

    def get_status(self, point1, point2, X, Xc):

        def get_d(x1, y1, x2, y2, x, y):
            return (x-x1)*(y2-y1) - (y-y1)*(x2-x1)

        x1, y1 = Xc
        x2, y2 = X

        X1, Y1 = point1
        X2, Y2 = point2

        D1 = get_d(x1, y1, x2, y2, X1, Y1)
        D2 = get_d(x1, y1, x2, y2, X2, Y2)

        if D1 * D2 > 0:
            return False
        
        else:
            return True

    def make_graph(self, k):
        """Create the graph based on k-nearest neighbours"""
        for vertex in self.graph.vertices:
            center = vertex.center
            nearest = self.graph.get_k_nearest_neighbours(vertex, k)[1:]

            current = nearest[0]

            # Make the connection to the first one
            self.graph.add_edge(vertex, current)

            i = 1
            j = 1
            while j < len(nearest) and i < k:
                next_node = nearest[j]

                # Calculate the angle between those nodes
                point1 = [current.state[0][0], current.state[1][0]]
                point2 = [next_node.state[0][0], next_node.state[1][0]]
                X = [vertex.state[0][0], vertex.state[1][0]]
                Xc = [center[0][0], center[1][0]]

                dot = self.get_status(point1, point2, X, Xc)
                if dot:
                    self.graph.add_edge(vertex, next_node)

                    current = next_node
                    i += 1
                
                j += 1
            
            if vertex in vertex.connections:
                raise Exception("Same Vertex")

        if self.SWITCH:
            # Make the switching points
            self.update_switching_points()

    def cost(self, node1: Node, node2: Node):
        """This function should calculate the cost of traversing from one node to another"""
        # Dist cost: Distance from node1 to node2
        dist_cost = Node.distance(node1.state, node2.state)
        traversal_penalty_cost = node2.cost
        
        # utility = self.gain_calculator.utility(node1, node2)
        utility = -100 if node2.isSwitching else 0

        return TraversalWithKeys.Key(
            traversal_penalty_cost,
            utility,
            dist_cost
        )

    def get_sorted(self, current, connections):
        """Sort the connections based on their cost"""
        cons = connections.copy()
        for i in range(len(cons)):
            for j in range(0, len(cons)-i-1):
                c1 = self.cost(current, cons[j])
                c2 = self.cost(current, cons[j+1])
                if c1 > c2:
                    cons[j], cons[j+1] = cons[j+1], cons[j]

        return cons

    def calculate_path_cost(self, path):
        cost = 0
        for i in range(len(path)-1):
            cost += np.linalg.norm(path[i].state - path[i+1].state)

        return cost

    def traverse(self, start: Node):
        """Main function that traverses the graph"""
        # 1. Set start to traversed and add a cost of 100 to it
        start.traversed = True
        start.cost += self.TRAVERSAL_PENALTY

        path = [start]

        # 2. Set the number of vertices traversed
        vertices_traversed = 1

        # 3. Set the current node to any of starts connections
        current = start

        i = 0
        while vertices_traversed < len(self.graph.vertices):
            neighbours = self.get_sorted(current, current.connections)
            connected = False

            if True:

                for neighbour in neighbours:
                    if not neighbour.traversed:
                        # Traverse to neighbour
                        neighbour.traversed = True
                        neighbour.cost += self.TRAVERSAL_PENALTY
                        connected = True
                        vertices_traversed += 1
                        path.append(neighbour)

                        # Set current to neighbour and traverse
                        current = neighbour
                        break

                if not connected:
                    # This will happen when all the neighbours have been traversed.
                    # If so select the first one [Least Cost] that is connectable.

                    current = neighbours[0]
                    neighbours[0].cost += self.TRAVERSAL_PENALTY
                    path.append(neighbours[0])

            i += 1
            if vertices_traversed == len(self.graph.vertices):
                if VERBOSE:
                    print("done")
                break
            if i == self.limit:
                if VERBOSE:
                    print("Exceeded limit")
                break
        return path


def test(circles, obstacles):
    from time import time
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

    return endTime, cost