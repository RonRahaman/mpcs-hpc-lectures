cudaEvent_t tick, tock;
cudaEventCreate(&tick);
cudaEventCreate(&tock);

cudaEventRecord(tick, 0);
kernel<<<griddim,blockdim>>> (foo, bar);
cudaEventRecord(tock, 0);

cudaEventSynchronize(tock);

float time;
cudaEventElapsedTime(&time, tick, tock);

cudaEventDestroy(tick);
cudaEventDestroy(tock);
