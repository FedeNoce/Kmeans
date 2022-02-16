import os

import sklearn.datasets as datasets
import pandas as pd
import matplotlib.pyplot as plt


def generate_dataset(points, n_features, centers, std, file_name, output_directory):
    data, __= datasets.make_blobs(n_samples=points, n_features=n_features, centers=centers, cluster_std=std, shuffle=True, random_state=1000)
    pd.DataFrame(data).to_csv(
        os.path.join(output_directory, file_name), header=False, index=False
    )
    pd.DataFrame(data).plot(kind='scatter', x='first', y='second', color='red')
    plt.show()

def main():


    DATASETS_DIR = 'datasets/'

    # two dimensional datasets

    generate_dataset(points=1000, n_features=2, centers=2, std=1, file_name=f'2D_data_3.csv', output_directory=DATASETS_DIR)


    # '''Datasets with an increasing number of points'''
    #
    # DATASETS_DIR = '../datasets/different_size/'
    #
    # # two dimensional datasets
    # for points in [100, 1000, 10000, 20000, 50000, 100000, 250000, 500000]:
    #     generate_dataset(points, n_features=2, centers=5, std=1, file_name=f'2D_data_{points}.csv', output_directory=DATASETS_DIR)
    #
    # # three dimensional datasets
    #
    # for points in [100, 1000, 10000, 20000, 50000, 100000, 250000, 500000]:
    #     generate_dataset(points, n_features=3, centers=5, std=1, file_name=f'3D_data_{points}.csv', output_directory=DATASETS_DIR)


if __name__ == '__main__':
    main()