__global__ void dot_kernel(const float * a, const float * b,
                           float *partial_c, int n) {
  // Allocate shared memory
  __shared__ float cache[block_dim];
  // Accumulate partial product in shared memory
  cache[threadIdx.x] = 0;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x;
       i < n;
       i += blockDim.x * gridDim.x) {
    cache[threadIdx.x] += a[i] * b[i];
  }
  __syncthreads();
  // ...
}
