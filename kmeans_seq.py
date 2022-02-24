#Kmeans 2D sequential implementation

import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

#Function for euclidean distance
def euclidean_distance_2D(x1, x2, y1, y2):
    return math.sqrt(((x1-y1) ** 2) + ((x2-y2) ** 2))

#Function for centroid assignment
def centroid_assg(points, centroids, points_assg):
    for i in range(n):
        distance = 100000
        for j in range(k):
            dist = euclidean_distance_2D(points[i, 0], points[i, 1], centroids[j, 0], centroids[j, 1])
            if dist < distance:
                distance = dist
                points_assg[i] = j

#Function for centroid update
def centroid_update(points, points_assgn):
    centroids_sum = np.zeros((k, 2))
    cluster_size = np.zeros(k)
    for i in range(n):
        clust_id = points_assgn[i]
        clust_id = int(clust_id)
        cluster_size[clust_id] = cluster_size[clust_id] + 1
        centroids_sum[clust_id, 0] += points[i, 0]
        centroids_sum[clust_id, 1] += points[i, 1]
    cluster_size = np.vstack((cluster_size, cluster_size))
    return centroids_sum/cluster_size.T


num_iter = 20
k = 4
n = 0

#Read the data
with open('datasets/2D_data_3.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = csv.reader(read_obj)
    # Iterate over each row in the csv using reader object
    x_coords = []
    y_coords = []
    for row in csv_reader:
        n = n+1
        x_coords.append(float(row[0]))
        y_coords.append(float(row[1]))

points = np.vstack((x_coords, y_coords))
points = points.T

#K-means iterations
centroids = np.zeros((k, 2))
points_assg = np.zeros((n))
for i in range(k):
    rand = random.randint(0, n)
    centroids[i, 0] = points[rand, 0]
    centroids[i, 1] = points[rand, 1]

start_time = time.time()
for i in range(num_iter):
    start_time_iter = time.time()
    centroid_assg(points, centroids, points_assg)
    centroids = centroid_update(points, points_assg)
    print("Seconds for iter " + str(i) + ': ' + str((time.time() - start_time_iter)))

print("Seconds for clustering: " + str((time.time() - start_time)))

plt.scatter(points[:, 0], points[:, 1], c=points_assg)
plt.show()