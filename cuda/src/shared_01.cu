my_kernel<<<grid_dim, block_dim, n * sizeof(float)>>>(arg1, arg2);