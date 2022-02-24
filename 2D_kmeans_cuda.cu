#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>


#define N 100000
#define K 4
#define MAX_ITER 50
#define TPB 256
#define EPSILON 0.00000001

__device__ float distance_2D(const float x1, const float x2, const float y1, const float y2)
{
    return sqrt(pow((x1-y1),2) + pow((x2-y2),2));
}


__global__ void kMeansClusterAssignment(const float *d_datapoints_x, const float *d_datapoints_y, int *d_clust_assn, const float *d_centroids_x, const float *d_centroids_y)
{
    //get idx for this datapoint
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx >= N) return;


    //find the closest centroid to this datapoint
    float min_dist = INFINITY;
    int closest_centroid = 0;

    for(int c = 0; c<K; ++c)

    {
        float dist = distance_2D(d_datapoints_x[idx], d_datapoints_y[idx], d_centroids_x[c], d_centroids_y[c]);

        if(dist < min_dist)
        {
            min_dist = dist;
            closest_centroid=c;
        }
    }

    //assign closest cluster id for this datapoint/thread
    d_clust_assn[idx]=closest_centroid;
}


__global__ void kMeansCentroidUpdate_Sum(const float *d_datapoints_x, const float *d_datapoints_y, const int *d_clust_assn, float *d_centroids_x, float *d_centroids_y, float *d_clust_sizes)
{

    //get idx of thread at grid level
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx >= N) return;

    //get idx of thread at the block level
    const int s_idx = threadIdx.x;

    //put the datapoints and corresponding cluster assignments in shared memory so that they can be summed by thread 0 later
    __shared__ float s_datapoints_x[TPB];
    s_datapoints_x[s_idx]= d_datapoints_x[idx];

    __shared__ float s_datapoints_y[TPB];
    s_datapoints_y[s_idx]= d_datapoints_y[idx];

    __shared__ int s_clust_assn[TPB];
    s_clust_assn[s_idx] = d_clust_assn[idx];

    __syncthreads();

    //it is the thread with idx 0 (in each block) that sums up all the values within the shared array for the block it is in
    if(s_idx==0)
    {
        float b_clust_datapoint_sums_x[K]={0};
        float b_clust_datapoint_sums_y[K]={0};
        float b_clust_sizes[K]={0};

        for(int j=0; j < blockDim.x; ++j)
        {
            int clust_id = s_clust_assn[j];
            b_clust_datapoint_sums_x[clust_id]+=s_datapoints_x[j];
            b_clust_datapoint_sums_y[clust_id]+=s_datapoints_y[j];
            b_clust_sizes[clust_id]+=1;
        }
        //Now we add the sums to the global centroids and add the counts to the global counts.
        for(int z=0; z < K; ++z)
        {
            atomicAdd(&d_centroids_x[z],b_clust_datapoint_sums_x[z]);
            atomicAdd(&d_centroids_y[z],b_clust_datapoint_sums_y[z]);
            atomicAdd(&d_clust_sizes[z],b_clust_sizes[z]);
        }
    }

    __syncthreads();

    //currently centroids are just sums, so divide by size to get actual centroids
    if(idx < K){
        d_centroids_x[idx] = d_centroids_x[idx]/d_clust_sizes[idx];
        d_centroids_y[idx] = d_centroids_y[idx]/d_clust_sizes[idx];
    }

}
__global__ void kMeansCentroidUpdate_Div(float *d_centroids_x, float *d_centroids_y, float *d_clust_sizes){
    int idx = threadIdx.x;
    d_centroids_x[idx] = d_centroids_x[idx]/d_clust_sizes[idx];
    d_centroids_y[idx] = d_centroids_y[idx]/d_clust_sizes[idx];
}

//Function to stop the algorithm when convergence is reached
int compareArrays(float a[], float b[], int n) {
    int i;
    for(i=0; i<n; i++){
        if(abs(a[i]-b[i]) > EPSILON)
            return 0;
    }
    return 1;
}


int main()
{
    srand(time(NULL));   // Initialization, should only be called once.
    FILE *fpt;
    //FILE *fpt_centroids;

    fpt = fopen("/home/federico/CLionProjects/kmeans_cuda/datasets/2D_data_3.csv", "r");
    //fpt = fopen("/home/federico/CLionProjects/kmeans_cuda/datasets/2D_data_uniform.csv", "r");
    //fpt_centroids = fopen("/home/federico/CLionProjects/kmeans_cuda/datasets/2D_data_3_centroids.csv", "r");

    //allocate memory on the device for the data points
    float *d_datapoints_x;
    float *d_datapoints_y;
    //allocate memory on the device for the cluster assignments
    int *d_clust_assn;
    //allocate memory on the device for the cluster centroids
    float *d_centroids_x;
    float *d_centroids_y;
    //allocate memory on the device for the cluster sizes
    float *d_clust_sizes;

    cudaMalloc(&d_datapoints_x, N*sizeof(float));
    cudaMalloc(&d_datapoints_y, N*sizeof(float));
    cudaMalloc(&d_clust_assn,N*sizeof(int));
    cudaMalloc(&d_centroids_x,K*sizeof(float));
    cudaMalloc(&d_centroids_y,K*sizeof(float));
    cudaMalloc(&d_clust_sizes,K*sizeof(float));

    //allocate memory for host
    float *h_centroids_x = (float*)malloc(K*sizeof(float));
    float *h_centroids_y = (float*)malloc(K*sizeof(float));
    float *h_datapoints_x = (float*)malloc(N*sizeof(float));
    float *h_datapoints_y = (float*)malloc(N*sizeof(float));
    float *h_current_centroids_x = (float*)malloc(K*sizeof(float));
    float *h_current_centroids_y = (float*)malloc(K*sizeof(float));
    int *h_clust_assn = (int*)malloc(N*sizeof(int));
    int *h_clust_sizes = (int*)malloc(K*sizeof(float));



    //initalize datapoints from csv
    printf("DataPoints: \n");
    for(int i=0;i<N;++i){
        fscanf(fpt,"%f,%f\n", &h_datapoints_x[i], &h_datapoints_y[i]);
        printf("(%f, %f) \n",  h_datapoints_x[i], h_datapoints_y[i]);
    }
    fclose(fpt);


    //initialize centroids, choose k-random points from datapoints
    printf("Clusters: \n");
    for(int i=0;i<K;++i){
        int r = rand() % N;
        h_centroids_x[i] = h_datapoints_x[r];
        h_centroids_y[i] = h_datapoints_y[r];
        h_current_centroids_x[i]=0.0;
        h_current_centroids_y[i]=0.0;
        printf("(%f, %f) \n",  h_centroids_x[i], h_centroids_y[i]);
        h_clust_sizes[i]=0;
    }



    //copy datapoints and all other data from host to device
    cudaMemcpy(d_centroids_x,h_centroids_x,K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids_y,h_centroids_y,K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_datapoints_x,h_datapoints_x,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_datapoints_y,h_datapoints_y,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_clust_sizes,h_clust_sizes,K*sizeof(float),cudaMemcpyHostToDevice);



    //Start time for clustering
    clock_t start = clock();
    int cur_iter = 0;

    while(cur_iter < MAX_ITER)
    {
        printf("Iter %d: \n",cur_iter);
        //Start time for iteration
        clock_t start_iter = clock();

        //Points assg
        kMeansClusterAssignment<<<(N+TPB-1)/TPB, TPB>>>(d_datapoints_x, d_datapoints_y, d_clust_assn, d_centroids_x, d_centroids_y);

        //copy new centroids back to host
        cudaMemcpy(h_centroids_x,d_centroids_x,K*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_centroids_y,d_centroids_y,K*sizeof(float),cudaMemcpyDeviceToHost);

        //copy assg back to host
        cudaMemcpy(h_clust_assn,d_clust_assn,N*sizeof(int),cudaMemcpyDeviceToHost);

        for(int i =0; i < K; i++){
            printf("C %d: (%f, %f)\n",i,h_centroids_x[i],h_centroids_y[i]);
        }
        //reset centroids and cluster sizes (will be updated in the next kernel)
        cudaMemset(d_centroids_x,0.0,K*sizeof(float));
        cudaMemset(d_centroids_y,0.0,K*sizeof(float));
        cudaMemset(d_clust_sizes,0,K*sizeof(int));

        //call centroid update
        kMeansCentroidUpdate_Sum<<<(N+TPB-1)/TPB, TPB>>>(d_datapoints_x, d_datapoints_y, d_clust_assn, d_centroids_x, d_centroids_y, d_clust_sizes);

        //kMeansCentroidUpdate_Div<<<1, K>>>(d_centroids_x, d_centroids_y, d_clust_sizes);

/*
        cudaMemcpy(h_current_centroids_x,d_centroids_x,K*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_current_centroids_y,d_centroids_y,K*sizeof(float),cudaMemcpyDeviceToHost);
        for(int i=0;i<K;++i){
            printf("h(%f, %f) \n",  h_current_centroids_x[i], h_current_centroids_y[i]);
            h_clust_sizes[i]=0;
        }
*/


        //Stop time for iteration
        clock_t end_iter = clock();
        float seconds_iter = (float)(end_iter - start_iter) / CLOCKS_PER_SEC;
        printf("Time for iter: %f\n", seconds_iter);

        //Compare the centroids for stop the clustering
        cudaMemcpy(h_current_centroids_x,d_centroids_x,K*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(h_current_centroids_y,d_centroids_y,K*sizeof(float),cudaMemcpyDeviceToHost);

        if(compareArrays(h_current_centroids_x, h_centroids_x, K) && compareArrays(h_current_centroids_y, h_centroids_y, K))
            break;


        cur_iter+=1;
    }


    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    printf("Time for clustering: %f\n", seconds);

    FILE *res;

    res = fopen("/home/federico/CLionProjects/kmeans_cuda/results/2D_data_3_results.csv", "w+");
    for(int i=0;i<N;i++){
        fprintf(res,"%d\n", h_clust_assn[i]);
    }

    cudaFree(d_datapoints_x);
    cudaFree(d_datapoints_y);
    cudaFree(d_clust_assn);
    cudaFree(d_centroids_x);
    cudaFree(d_centroids_y);
    cudaFree(d_clust_sizes);

    free(h_centroids_x);
    free(h_centroids_y);
    free(h_current_centroids_x);
    free(h_current_centroids_y);
    free(h_datapoints_x);
    free(h_datapoints_y);
    free(h_clust_assn);
    free(h_clust_sizes);

    return 0;
}

