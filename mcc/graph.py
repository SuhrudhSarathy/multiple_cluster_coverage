import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

class Node:
    """Template Node class"""
    def __init__(self, state: np.ndarray, center: np.ndarray=None):
        self.state = state
        self.center = center
        if self.center is None:
            self.center = np.zeros_like(self.state)
        self.connections: list = []
        self.costs: list[float] = []

        self.traversed = False
        self.traversed_count = 0
        self.cost = 0

        self.isSwitching = False

        self.radius = None

    @staticmethod
    def distance(node1, node2):
        return np.linalg.norm(node1-node2)
    
    def add_connection(self, node, cost: float = 0.0):
        """This is helpful when you want to add a connection with a cost"""
        self.connections.append(node)
        self.costs.append(cost)

    def remove_connection(self, node):
        """Removes the connection between two nodes"""
        i = self.connections.index(node)
        self.costs.pop(i)
        self.connections.remove(node)

    def __len__(self):
        """Returns the length of connections of a node"""
        return len(self.connections)

    def __str__(self):
        """Returns the string format to print"""
        return str(self.state)

    def __eq__(self, __o: object):
        return np.linalg.norm(self.state-__o.state) < 1e-4

class Graph:
    """Basic Graph Data Structure"""
    def __init__(self):
        self.vertices: list[Node] = []

        # Count of the edges and vertices
        self.vertex_count: int = 0
        self.edge_count: int = 0

        self.edges = []

    def add_vertex(self, node: Node):
        """Adds a vertex to the graph.
        This does not check if the vertex added is already present in the graph or not
        """
        self.vertices.append(node)
        self.vertex_count += 1

    def add_edge(self, node1: Node, node2: Node, cost: float=0.0):
        """Adds and undirected edge between the node1 and node2"""
        if [node1, node2]  not in self.edges and [node2, node1] not in self.edges:
            # this will make sure that this only happens once
            self.edges.append([node1, node2])
            node1.add_connection(node2, cost)
            node2.add_connection(node1, cost)
            # Add this edge into the edges list also

            self.edge_count += 1

      
    def remove_vertex(self, node: Node):
        """Removes a vertex from the graph.
        Also iterates through all the connections and remove the node
        """
        for neighbour in node.connections:
            neighbour.remove_connection(node)
            self.edge_count -= 1
        self.vertices.remove(node)
        self.vertex_count -= 1

    def remove_edge(self, node1: Node, node2: Node):
        """Removes the edge between the given nodes"""
        if [node1, node2] in self.edges or [node2, node1] in self.edges:
            node1.remove_connection(node2)
            node2.remove_connection(node1)

            self.edge_count -= 1
        else:
            print("Edge not in Graph")

    def construct_kd_tree(self):
        pts = []
        for vertex in self.vertices:
            pts.append(vertex.state.reshape(vertex.state.shape[0],))
        pts = np.array(pts)

        self.tree = KDTree(pts)

    # Useful functions
    def get_neighbours(self, node: Node):
        """Returns the neighbours of a given node"""
        return node.connections

    def get_k_nearest_neighbours(self, node: Node, k: int=1):
        # definetely optimise this. This is the bottleneck
        if type(node) == Node:
            dist, ind = self.tree.query(node.state.reshape(1, -1), k)
        else:
            dist, ind = self.tree.query(node, k)

        return [self.vertices[i] for i in ind[0]]

    def plot_graph(self, ax: plt.Axes):
        X = []
        Y = []
        for vertex in self.vertices:
            x, y = vertex.state[0][0], vertex.state[1][0]
            connections = self.get_neighbours(vertex)

            for connection in connections:
                x_, y_ = connection.state[0][0], connection.state[1][0]

                ax.plot([x, x_], [y, y_], color="red", alpha=0.25)

            X.append(x)
            Y.append(y)

        ax.scatter(X, Y, color="green", s=15)

    def plot_vertices(self, ax: plt.Axes, vertices, color="green"):
        X = []
        Y = []

        for vertex in vertices:
            X.append(vertex.state[0][0])
            Y.append(vertex.state[1][0])

        plt.plot(X, Y, color=color, linewidth=2)

    def __len__(self):
        return self.vertex_count

    def return_kruskal_graph(self):
        kruskal_graph = KruskalGraph(self.vertex_count)
        for edge in self.edges:
            e1 = self.vertices.index(edge[0])
            e2 = self.vertices.index(edge[1])

            kruskal_graph.add_edge(e1, e2, Node.distance(edge[0].state, edge[1].state))

        kruskal_graph.kruskal()

        new_graph = Graph()

        for [u, v, cost] in kruskal_graph.result:
            n1 = Node(self.vertices[u].state.copy())
            n2 = Node(self.vertices[v].state.copy())

            n1.radius = self.vertices[u].radius
            n2.radius = self.vertices[v].radius

            new_graph.add_vertex(n1)
            new_graph.add_vertex(n2)

            new_graph.add_edge(n1, n2, cost)

        return new_graph

class KruskalGraph:
    def __init__(self, vertex):
        self.V = vertex
        self.graph = []
 
    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])
 
 
    def search(self, parent, i):
        if parent[i] == i:
            return i
        return self.search(parent, parent[i])
 
    def apply_union(self, parent, rank, x, y):
        xroot = self.search(parent, x)
        yroot = self.search(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1
 
  
    def kruskal(self):
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.search(parent, u)
            y = self.search(parent, v)
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.apply_union(parent, rank, x, y)
        self.result = result

if __name__ == "__main__":
    nodes = [
        [5, 5],
        [0, 0],
        [3, 0],
        [5, 0],
        [0, 5],
        [3, 5],
        
    ]

    vertices = [Node(np.array(n).reshape(-1, 1)) for n in nodes]

    graph = Graph()
    for vertex in vertices:
        graph.add_vertex(vertex)

    for i, vertex in enumerate(vertices):
        for j, vertex2 in enumerate(vertices[i+1:]):
            graph.add_edge(vertex, vertex2, Node.distance(vertex.state, vertex2.state))

    new_graph = graph.return_kruskal_graph()

    fig, ax = plt.subplots(figsize=(6, 6))
    new_graph.plot_graph(ax)

    plt.show()