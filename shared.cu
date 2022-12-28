#include <stdio.h>

#include <cuda.h>

#define N 1000 // size of array

global void bubbleSort(float * a) {
    shared float temp[N];

    // each thread reads one element from global memory
    int i = threadIdx.x;
    temp[i] = a[i];

    __syncthreads(); // synchronize threads to make sure all reads are completed

    // perform bubble sort algorithm on shared memory
    for (int j = 0; j < N - 1; j++) {
        if (temp[j] > temp[j + 1]) {
            float t = temp[j];
            temp[j] = temp[j + 1];
            temp[j + 1] = t;
        }
    }

    __syncthreads(); // synchronize threads again before writing back to global memory

    // each thread writes one element back to global memory
    a[i] = temp[i];

}

int main() {
    float a[N], b[N];

    // initialize array 'a' with random values
    for (int i = 0; i < N; i++)
        a[i] = (float) rand() / RAND_MAX;

    // copy array 'a' to device memory
    float * dev_a;
    cudaMalloc((void ** ) & dev_a, N * sizeof(float));
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

    // launch kernel on the GPU
    bubbleSort << < 1, N >>> (dev_a);

    // copy sorted array back to host memory
    cudaMemcpy(b, dev_a, N * sizeof(float), cudaMemcpyDeviceToHost);

    // print sorted array
    for (int i = 0; i < N; i++)
        printf("%f ", b[i]);
    printf("\n");

    cudaFree(dev_a);

    return 0;

    // // Initialize an array of integers to sort
    // int data[] = {959, 789, 917, 499, 834, 594, 991, 668, 671, 861, 301, 971, 189, 629, 870, 742, 665, 879, 489, 734, 479, 380, 517, 964, 173, 773, 479,
    //  842, 776, 835, 479, 802, 987, 894, 
    // 463, 579, 964, 847, 868, 574, 806, 493, 678, 891, 977, 629, 863, 574, 967, 684, 616, 579, 493, 715, 758, 879, 673, 838, 673, 614, 758};
}