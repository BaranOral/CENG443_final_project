#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h>

#define N 10240  // size of array
#define BLOCK_SIZE 1024 

__global__ void myBubbleSort(float *dev_a) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // each thread reads one element from global memory
    float temp = dev_a[gid];
    
    // allocate dynamically-sized shared memory array
    __shared__ float chunk[BLOCK_SIZE];
    
    // copy elements from global memory to shared memory
    chunk[tid] = temp;
    
    // synchronize threads to make sure all reads are completed
    __syncthreads();
    
    // perform bubble sort algorithm on shared memory
    for (int i = 0; i < BLOCK_SIZE; i++) {
        // modify inner loop to only process necessary elements
        for (int j = tid; j < BLOCK_SIZE - i - 1; j += blockDim.x) {
            if (chunk[j] > chunk[j + 1]) {
                float t = chunk[j];
                chunk[j] = chunk[j + 1];
                chunk[j + 1] = t;
            }
        }
        __syncthreads();
    }

    // write sorted elements back to global memory
    if (gid < N) {
        dev_a[gid] = chunk[tid];
    }

}
__global__ void merge(float *dev_a, int chunkSize) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;


    // allocate shared memory for inter-block communication
    __shared__ float temp[BLOCK_SIZE];

    // each thread reads one element from global memory
    temp[tid] = dev_a[gid];

    // synchronize threads to make sure all reads are completed
    __syncthreads();

    // merge elements in shared memory
    for (int i = 1; i < chunkSize; i *= 2) {
        if ((tid % (2 * i)) == 0) {
            if (temp[tid] > temp[tid + i]) {
                float t = temp[tid];
                temp[tid] = temp[tid + i];
                temp[tid + i] = t;
            }
        }
        __syncthreads();
    }

    // write merged elements back to global memory
    if (gid < chunkSize) {
        dev_a[gid] = temp[tid];
    }

}

int main() {
    clock_t start, end;
    double time_used;
    float * a, * b;
    a = (float *) malloc(N * sizeof(float));
    b = (float *) malloc(N * sizeof(float));

    // initialize array 'a' with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++)
        a[i] = (float) drand48();

    // copy array 'a' to device memory
    float * dev_a;
    start = clock();
    cudaError_t error = cudaMalloc((void ** ) & dev_a, N * sizeof(float));
    if (error != cudaSuccess) {
        printf("Error allocating device memory: %s\n", cudaGetErrorString(error));
        return 1;
    }

    error = cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error copying host to device: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // number of chunks
    int chunkSize = BLOCK_SIZE;
    int blocks = ceil(N/BLOCK_SIZE);

    for (int i = 0; i < blocks; i++) {
        // launch kernel to sort chunk i
        myBubbleSort<<<blocks, BLOCK_SIZE>>>(dev_a + i * chunkSize);

        // launch kernel to merge chunk i with the next chunk (i+1)
        merge<<<blocks, BLOCK_SIZE>>>(dev_a + i * chunkSize, chunkSize);
    }
    cudaDeviceSynchronize();

    // copy sorted array back to host memory
    error = cudaMemcpy(b, dev_a, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error copying device to host: %s\n", cudaGetErrorString(error));
        return 1;
    }
   
    end = clock();
    time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("%lf is total compile time on GPU. \n", time_used);


   
    // free device and host memory
    cudaFree(dev_a);
    free(a);
    free(b);

    return 0;
}
