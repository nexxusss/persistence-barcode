import numpy as np
import matplotlib.pyplot as plt
from math import dist

def generate_points_in_annulus(num_points, inner_radius, outer_radius, seed=1):

    np.random.seed(seed)

    # Generate random angles
    theta = 2 * np.pi * np.random.rand(num_points)

    # Generate random radii within the annulus
    r = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, num_points))

    # Convert polar coordinates to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Combine x and y to get points in the annulus
    points = np.column_stack((x, y))
    
    return points

def generate_circles(points, radius):
    # plot a growing cirlce around for each point
    circles = []
    for point in points:
        circle = plt.Circle((point[0], point[1]), radius, color='blue', fill=False)
        circles.append([circle, radius])


    return circles

def get_intersections(points, circles):
    # find overlapping circles/intersections and link vertices with edges
    edges = []

    for i in range(len(points)):
        #check every other points
        point1 = points[i]
        for j in range(i+1, len(points)):
            point2 = points[j]

            # get diameters of circles around point 1 and 2
            d1 = circles[i][1]
            d2 = circles[j][1]

            # compute the distance between two points
            x = dist(point1, point2)

            #check if the sum of the radius of pt1 and pt2 is greater or equal to 
            #the distance between the points

            if d1 + d2 >= x:
                # Check if the edge already exists in the list
                if not any(np.array_equal(edge, [point1, point2]) or np.array_equal(edge, [point2, point1]) for edge in edges):
                    edges.append([point1, point2])

    return edges

def plot_edges(edges):
    for edge in edges:
        plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color='green')

num_points = 15
inner_radius = 1
outer_radius = 2
radius = 0.8

# Generate points in the annulus
points = generate_points_in_annulus(num_points, inner_radius, outer_radius, seed=2)
circles = generate_circles(points, radius=radius) # <-- returns an array with the circle object and radius
edges  = get_intersections(points, circles)


# Plot the points in the annulus
fig, ax = plt.subplots()
for circle in circles:
    ax.add_patch(circle[0])

plt.scatter(points[:, 0], points[:, 1], c='red', marker='o')
plot_edges(edges)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Points in Annulus')
plt.axis('equal')
plt.show()