import os

import sklearn.datasets as datasets
import pandas as pd




def generate_dataset(points, n_features, centers, std, file_name, centroids_file_name, output_directory):
    data, __, centroids = datasets.make_blobs(n_samples=points, n_features=n_features, centers=centers, cluster_std=std, shuffle=True, random_state=1000, return_centers=True)
    pd.DataFrame(data).to_csv(
        os.path.join(output_directory, file_name), header=False, index=False
    )
    pd.DataFrame(centroids).to_csv(
        os.path.join(output_directory, centroids_file_name), header=False, index=False
    )

def main():


    DATASETS_DIR = 'datasets/'

    # two dimensional datasets
    for p in [100, 1000, 10000, 20000, 50000, 100000, 250000, 500000]:
        generate_dataset(points=p, n_features=2, centers=5, std=0.3, file_name=f'2D_data_{p}.csv', centroids_file_name=f'2D_data_3_centroids.csv', output_directory=DATASETS_DIR)
    #
    # # three dimensional datasets
    #
    # for points in [100, 1000, 10000, 20000, 50000, 100000, 250000, 500000]:
    #     generate_dataset(points, n_features=3, centers=5, std=1, file_name=f'3D_data_{points}.csv', output_directory=DATASETS_DIR)


if __name__ == '__main__':
    main()