  int grid_dim = (n + block_dim-1) / block_dim;
  dot_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_partial_c, n);