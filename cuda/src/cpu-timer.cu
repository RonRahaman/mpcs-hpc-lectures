timespec_t start, kernel_end, memcpy_end;

clock_gettime(CLOCK_MONOTONIC, &start);
my_awesome_kernel<<<griddim,blockdim>>>(foo, bar);
clock_gettime(CLOCK_MONOTONIC, &kernel_end);

cudaMemcpy(A, d_A, n * sizeof(float), cudaMemcpyDeviceToHost);
clock_gettime(CLOCK_MONOTONIC, &memcpy_end);