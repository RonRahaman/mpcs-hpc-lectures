int main() {
  // ...
  dot_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_partial_c, n);
  cudaMemcpy(partial_c, d_partial_c, grid_dim * sizeof(float),
             cudaMemcpyDeviceToHost);

  // One last reduction on host!
  float c;
  for (int i =0 ; i < grid_dim; ++i) {
    c += partial_c[i];
  }
}

