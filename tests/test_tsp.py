
import numpy as np
import matplotlib.pyplot as plt


from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# import shapely to perform collision aware selection
from shapely.geometry import LineString, Point

NOISE = 0.01
COUNT = 25

def generate_circle(n, m, r, X, Y):
    
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
    def generate_cirlces_in_aisle(n, m, r, bounds_x, bounds_y, ADD_NOISE=True):
        X = np.linspace(0, bounds_x, n)
        Y = np.linspace(0, bounds_y, m)

        circles = []

        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                if ADD_NOISE:
                    np.random.seed(i)
                    x_ = np.random.rand()
                    np.random.seed(j)
                    y_ = np.random.randn()
                    np.random.seed(np.random.randint(1000, 10000))
                    circles.append(
                        [[x+np.random.rand(), y+np.random.randn()], r + np.random.randn() * 0.01]
                    )
                else:
                    circles.append(
                        [[x, y], r]
                    )
        return circles

    def generate_obstacles(circles):
        obstacles = []
        for circle in circles:
            obstacles.append([circle[0], circle[1]*0.75])

        return obstacles
    def calculate_min_cost(circles):
        perimeters = 0

        for circle in circles:
            perimeters += 2*np.pi*circle[1]

        distances = 0
        for i in range(len(circles)):
            j = (i+1)%len(circles)

            c1, r1 = np.array(circles[i][0]), circles[i][1]
            c2, r2  = np.array(circles[j][0]), circles[j][1]

            distances += np.linalg.norm(c1-c2) - r1 - r2

        return perimeters + 2 * distances

    # 2 Circles
    circles = generate_cirlces_in_aisle(n, m, r, X, Y)
    obstacles = generate_obstacles(circles)

    return circles, obstacles

def get_locations_in_circle(circle, X):

    center, r = circle

    for theta in np.linspace(-np.pi, np.pi, COUNT):
        x = center[0] + r * np.cos(theta) + np.random.randn() * NOISE
        y = center[1] + r * np.sin(theta) + np.random.randn() * NOISE

        X.append([x, y])


# ALL TSP Functions
def get_distance_matrix(points, circles):

    def collision(v1, v2, circles):

        line = LineString([v1, v2])
        for circle in circles:
            if circle.intersects(line):
                return True
            
        else:
            return False

    def distance(p1, p2):
        # This should return an integer. Hence multiply this by 100 and then round it off
        return int(100*np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))
    data = {}
    distances = []

    for i, vertex in enumerate(points):
        distances.append([])
        for vertex2 in points:
            # Check for collision and then add them to the list
            is_collision = collision(vertex, vertex2, circles)

            if not is_collision:
                distances[i].append(distance(vertex, vertex2))
            else:
                distances[i].append(10000)
    
    data["distance_matrix"] = distances

    data["num_vehicles"] = 1
    # you can change the depo later. For now, let it be one
    data["depot"] = 10

    return data

def print_solution(manager, routing, solution):
    """Prints solution on console."""
    # print('Objective: {} miles'.format(solution.ObjectiveValue()/100))
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    # print(plan_output)
    plan_output += 'Route distance: {}miles\n'.format(route_distance)

def get_routes(solution, routing, manager):
  """Get vehicle routes from a solution and store them in an array."""
  # Get vehicle routes and store them in a two dimensional array whose
  # i,j entry is the jth location visited by vehicle i along its route.
  routes = []
  for route_nbr in range(routing.vehicles()):
    index = routing.Start(route_nbr)
    route = [manager.IndexToNode(index)]
    while not routing.IsEnd(index):
      index = solution.Value(routing.NextVar(index))
      route.append(manager.IndexToNode(index))
    routes.append(route)
  return routes


def tsp(points, circles, PRINT=False):


    circle_objects = [Point(c[0][0], c[0][1]).buffer(c[1]*0.85) for c in circles]


    # Full TSP functions now
    data = get_distance_matrix(points, circle_objects)
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]


    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    # if solution:
    #     print(f"Cost of Traversal: {solution.ObjectiveValue()/100}")
    # if solution and PRINT:
    #     print_solution(manager, routing, solution)

    routes = get_routes(solution, routing, manager)

    return routes[0], solution.ObjectiveValue()/100


def plot_route(route, points):
    path = []
    for node in route:
        path.append(points[node])

    fig, ax = plt.subplots()
    ax.scatter([p[0] for p in path], [p[1] for p in path], color="red")
    ax.plot([p[0] for p in path], [p[1] for p in path], color="green")

    plt.show()

def animate_route(route, points):
    fig, ax = plt.subplots()
    path = []
    for node in route:
        path.append(points[node])
    for i in range(len(path)):
        ax.scatter([p[0] for p in path], [p[1] for p in path], color="red")
        ax.plot([p[0] for p in path[0:i+1]], [p[1] for p in path[0:i+1]], color="green")

        ax.scatter(path[i][0], path[i][1], marker="*", s=75, color="black")

        # plt.savefig(f"./images/{i}.png")
        plt.pause(0.0001)
        plt.cla()


def test(circles):
    from time import time
    p1 = []
    for circle in circles:
        get_locations_in_circle(circle, p1)
    start = time()
    route, cost = tsp(p1, circles=circles, PRINT=False)
    end = time() - start

    return end, cost

if __name__ == "__main__":
    from time import time
    p1 = []
    circles, obstacles = generate_circle(2, 2, 2, 8, 8)
    
    for circle in circles:
        get_locations_in_circle(circle, p1)
    start = time()
    route, cost = tsp(p1, circles, False)
    end = time() - start

    print(end, cost)
    animate_route(route, p1)
