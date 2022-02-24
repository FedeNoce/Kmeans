cmake_minimum_required(VERSION 3.20)
project(kmeans CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")

set(SOURCE_FILES Kmeans_openMp.cpp Point.h Cluster.h)

add_executable(kmeans_cuda 2D_kmeans_cuda.cu)

set_target_properties(kmeans_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(kmeansOpenMP Kmeans_openMp.cpp Point.h Cluster.h)

