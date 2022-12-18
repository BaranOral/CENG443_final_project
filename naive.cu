#include <cuda.h>
#include <stdio.h>
#define N 10

__global__ void bubbleSort(float *d_arr) {
  // Get the index of the current element to be sorted
  int i = threadIdx.x;

  // Compare the current element with the next element
  if (d_arr[i] > d_arr[i + 1]) {
    // Swap the elements if the current element is greater than the next
    float temp = d_arr[i];
    d_arr[i] = d_arr[i + 1];
    d_arr[i + 1] = temp;
  }
}

__global__ void selectionSort(float *d_arr) {
  // Get the index of the current element to be sorted
  int i = threadIdx.x;

  // Find the minimum element in the array
  int minIndex = i;
  for (int j = i + 1; j < N; j++) {
    if (d_arr[j] < d_arr[minIndex]) {
      minIndex = j;
    }
  }

  // Swap the current element with the minimum element
  float temp = d_arr[i];
  d_arr[i] = d_arr[minIndex];
  d_arr[minIndex] = temp;
}

__global__ void insertionSort(float *d_arr) {
  int j;
  float temp;

  // Get the index of the current thread
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= N) {
    return;  // Out of bounds, return early
  }

  temp = d_arr[idx];
  j = idx - 1;
  while (j >= 0 && d_arr[j] > temp) {
    d_arr[j + 1] = d_arr[j];
    j--;
  }
  d_arr[j + 1] = temp;
}

__host__ float* generateRandomElements(){
    float constant = 1.0f;
    float *arr;
    arr = (float*)malloc(N*sizeof(float));  
    if ( arr == NULL ){
        printf("Run out of memmory!\n");
        exit(1);
    }

    for (int i = 0; i<N; i++){
          arr[i] = ((float)rand()/RAND_MAX)* constant; //generate random float element for array
          printf("index %d: %f | ", i , arr[i]);
          
    }
    return arr;
}

void printArray(float *array, int size){
    // Iterate over the elements of the array
    for (int i = 0; i < size; i++)
    {
        // Print the element
        printf("%f ", array[i]);
}
     printf("\n");

}


int main(void) {
  // Create host array
  


  float *h_arr, *d_arr;

  h_arr = generateRandomElements();

  // Allocate memory on the device
  cudaMalloc(&d_arr, N * sizeof(float));

  // Copy host array to device
  cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch bubbleSort kernel
  insertionSort<<<1, N>>>(d_arr);

  // Copy the sorted array back to the host
  cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);

  printArray(h_arr, N);
 
  // Free memory on the device
  cudaFree(d_arr);

}
