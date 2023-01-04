#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h>

#define N 10240  // size of array
#define BLOCK_SIZE 1024 

__global__ void chunkedBubbleSort(float *dev_a) {
    // Compute the global index for each thread
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Divide the array into chunks of size BLOCK_SIZE
    int chunkSize = BLOCK_SIZE;
    int chunkStart = gid * chunkSize;
    int chunkEnd = chunkStart + chunkSize - 1;
    
    // Make sure the chunk doesn't go out of bounds
    if (chunkEnd >= N) {
        chunkEnd = N - 1;
    }
    
    // Sort the chunk using bubble sort
    for (int i = chunkStart; i <= chunkEnd; i++) {
        for (int j = chunkStart; j < chunkEnd - i + chunkStart; j++) {
            if (dev_a[j] > dev_a[j + 1]) {
                float temp = dev_a[j];
                dev_a[j] = dev_a[j + 1];
                dev_a[j + 1] = temp;
            }
        }
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

    // launch kernel on the GPU
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
 
    int chunkSize = 512;
    chunkedBubbleSort<<<numBlocks, BLOCK_SIZE>>>(dev_a);

    
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
