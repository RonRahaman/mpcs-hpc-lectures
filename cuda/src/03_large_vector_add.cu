// Adapted from "CUDA by Example"

#include <stdlib.h>
#include <stdio.h>

__global__ void vec_add(const int *a, const int *b, int *c, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
      i < n; 
      i += gridDim.x * blockDim.x) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int n = 1024;

  int *a = (int *) malloc(n * sizeof(int));
  int *b = (int *) malloc(n * sizeof(int));
  int *c = (int *) malloc(n * sizeof(int));

  int *d_a, *d_b, *d_c;
  cudaMalloc((void **) &d_a, n * sizeof(int));
  cudaMalloc((void **) &d_b, n * sizeof(int));
  cudaMalloc((void **) &d_c, n * sizeof(int));

  for (int i=0; i < n; ++i) {
    a[i] = -i;
    b[i] = i * i;
  }

  cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

  vec_add<<<4,16>>>(d_a, d_b, d_c, n);

  cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

  int retval = 0;
  for (int i=0; i < n; ++i) {
    if (a[i] + b[i] != c[i]) {
      printf("Error: %d + %d = %d\n", a[i], b[i], c[i]);
      retval = 1;
      break;
    }
  }
  if (retval == 0) {
    printf("Success!\n");
  }

  free(a);
  free(b);
  free(c);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return retval;
}
