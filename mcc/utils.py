import numpy as np
from mcc.parameters import *

def generate_obstacles(circles):
        obstacles = []
        for circle in circles:
            obstacles.append([circle[0], circle[1]*0.75])

        return obstacles

def generate_circle(n, m, r):
    
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

        for x in X:
            for y in Y:
                if ADD_NOISE:
                    circles.append(
                        [[x + np.random.randn() * 0.75 , y + np.random.randn() * 0.75], r + np.random.randn() * 0.01]
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
    X = n * 2*r + (n-1) *2*r
    Y = m * 2*r + (m-1) *2*r
    circles = generate_cirlces_in_aisle(n, m, r, X, Y)
    obstacles = generate_obstacles(circles)

    return circles, obstacles