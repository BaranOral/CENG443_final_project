#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h>

#define N 10240 // size of array

// kernel function to perform bubble sort on the GPU
__global__ void paperBubbleSort(float * a) {
    int idx = threadIdx.x;
    // perform bubble sort algorithm on global memory
    for (int i = idx; i < N; i++) {
        for (int j = 0; j < N - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                float t = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }
        }
    }
}

__host__ void standartBubbleSort(float * a){
    for (int passnum = N-1; passnum > 0; passnum--) {
        for (int i = 0; i < passnum; i++) {
            if (a[i] > a[i+1]) {
                float temp = a[i];
                a[i] = a[i+1];
                a[i+1] = temp;
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
   

    // record start event

    dim3 threads(1024);
    dim3 blocks(ceil(N /1024));

    paperBubbleSort <<< blocks, threads >>> (dev_a);
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

   clock_t startStandart, endStandart;
   double time_used_Standart;
   startStandart = clock();

   standartBubbleSort(a);
   endStandart = clock();
   time_used_Standart = ((double) (endStandart - startStandart)) / CLOCKS_PER_SEC * 100;
   printf("%lf is total compile time on CPU. \n", time_used_Standart);


    // free device and host memory
    cudaFree(dev_a);
    free(a);
    free(b);

    return 0;
}
