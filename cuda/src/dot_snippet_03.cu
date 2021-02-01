__global__ void dot_kernel(const float * a, const float * b,
                           float *partial_c, int n) {
  // ...
  // Parallel reduction.  At the end, cache[0] will have result
  for (int stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      cache[threadIdx.x] += cache[threadIdx.x + stride];
    }
    __syncthreads();
  }
  // Each block puts its result in global memory
  if (threadIdx.x == 0) {
    partial_c[blockIdx.x] = cache[0];
  }
}
