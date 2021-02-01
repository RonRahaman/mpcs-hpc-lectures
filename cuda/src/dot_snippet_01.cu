const int block_dim = 256;

int main() {
  int n = pow(2, 27);
  int grid_dim = (n + block_dim-1) / block_dim;

  float *a = (float *) malloc(n * sizeof(float));
  float *b = (float *) malloc(n * sizeof(float));
  float *partial_c = (float *) malloc(grid_dim * sizeof(float));

  float *d_a, *d_b, *d_partial_c;
  cudaMalloc((void **) &d_a, n * sizeof(float));
  cudaMalloc((void **) &d_b, n * sizeof(float));
  cudaMalloc((void **) &d_partial_c, grid_dim * sizeof(float));
  // ...
}
