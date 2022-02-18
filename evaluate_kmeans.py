import csv
import numpy as np
import matplotlib.pyplot as plt

with open('datasets/2D_data_3.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = csv.reader(read_obj)
    # Iterate over each row in the csv using reader object
    x_coords = []
    y_coords = []
    for row in csv_reader:
        x_coords.append(float(row[0]))
        y_coords.append(float(row[1]))

with open('results/2D_data_3_results.csv', 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    points_assn = []
    for row in csv_reader:
        points_assn.append(int(row[0]))


x_coords = np.array((x_coords))
y_coords = np.array((y_coords))
points_assn = np.array((points_assn))

plt.scatter(x_coords, y_coords, c=points_assn)
plt.show()