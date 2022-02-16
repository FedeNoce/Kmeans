cmake_minimum_required(VERSION 3.20)
project(kmeans CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(kmeans main.cu)

set_target_properties(kmeans PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
