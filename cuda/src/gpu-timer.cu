cudaEvent_t tick, tock;
cudaEventCreate(&tick);
cudaEventCreate(&tock);

cudaEventRecord(tick, 0);
kernel<<<grid,threads>>> (arg1, arg2);
cudaEventRecord(tock, 0);

cudaEventSynchronize(tock);

float time;
cudaEventElapsedTime(&time, tick, tock);

cudaEventDestroy(tick);
cudaEventDestroy(tock);
