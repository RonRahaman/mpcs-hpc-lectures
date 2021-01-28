#pragma omp parallel for collapse(2)
for (int k = 0; k < n; ++k)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      C.at[i][j] += A.at[i][k] * B.at[k][j];
