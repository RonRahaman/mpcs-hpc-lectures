timespec_t start, kernel_end, memcpy_end;

clock_gettime(CLOCK_MONOTONIC, &start);
kernel<<<grid,threads>>>(arg1, arg2);
clock_gettime(CLOCK_MONOTONIC, &kernel_end);

cudaMemcpy(A, d_A, n * sizeof(float), cudaMemcpyDeviceToHost);
clock_gettime(CLOCK_MONOTONIC, &memcpy_end);