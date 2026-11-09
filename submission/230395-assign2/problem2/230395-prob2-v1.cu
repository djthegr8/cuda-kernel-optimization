#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#define NSEC_SEC_MUL (1.0e9)
#ifndef BLOCK
#define BLOCK 256
#endif
#define CAPACITY (1LL << 20)  // max result rows held on the device per launch

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),  \
              __FILE__, __LINE__);                                           \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

struct timespec begin_grid, end_main;

// to store values of disp.txt: 10 rows of (c_i1..c_i10, d_i, ey_i)
double a[120];

// to store values of grid.txt: 10 rows of (start, end, step)
double b[30];

// problem data in constant memory (same for all threads)
__constant__ double k_c[10][10];
__constant__ double k_d[10];
__constant__ double k_e[10];
__constant__ double k_start[10];
__constant__ double k_step[10];
__constant__ long long k_s[10];

// One thread = one grid point. n is decoded in mixed radix with r10 as the
// least significant digit, so the order matches the CPU loop nest.
__global__ void gridkernel(long long offset, long long count,
                           unsigned long long* nhits, double* out,
                           long long capacity) {
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count) return;
  long long n = offset + t;

  double x[10];
  for (int k = 9; k >= 0; --k) {
    long long r = n % k_s[k];
    n /= k_s[k];
    x[k] = k_start[k] + r * k_step[k];
  }

  // constraints: same summation order as the CPU code
  for (int i = 0; i < 10; ++i) {
    double sum = k_c[i][0] * x[0];
    for (int k = 1; k < 10; ++k) sum += k_c[i][k] * x[k];
    double q = fabs(sum - k_d[i]);
    if (!(q <= k_e[i])) return;
  }

  unsigned long long slot = atomicAdd(nhits, 1ULL);
  if (slot < (unsigned long long)capacity) {
    for (int k = 0; k < 10; ++k) out[slot * 10 + k] = x[k];
  }
}

int main() {
  int i, j;

  // read disp file
  FILE* fp = fopen("./disp.txt", "r");
  if (fp == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }
  for (i = 0; i < 120; ++i) {
    if (fscanf(fp, "%lf", &a[i]) != 1) {
      printf("Error: fscanf failed while reading disp.txt\n");
      exit(EXIT_FAILURE);
    }
  }
  fclose(fp);

  // read grid file
  FILE* fpq = fopen("./grid.txt", "r");
  if (fpq == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }
  for (j = 0; j < 30; ++j) {
    if (fscanf(fpq, "%lf", &b[j]) != 1) {
      printf("Error: fscanf failed while reading grid.txt\n");
      exit(EXIT_FAILURE);
    }
  }
  fclose(fpq);

  double kk = 0.3;

  // host-side setup
  double h_c[10][10], h_d[10], h_e[10];
  for (i = 0; i < 10; ++i) {
    for (j = 0; j < 10; ++j) h_c[i][j] = a[12 * i + j];
    h_d[i] = a[12 * i + 10];
    h_e[i] = kk * a[12 * i + 11];
  }

  double h_start[10], h_step[10];
  long long h_s[10];
  long long total = 1;
  for (int k = 0; k < 10; ++k) {
    h_start[k] = b[3 * k];
    double end = b[3 * k + 1];
    h_step[k] = b[3 * k + 2];
    long long sk = (long long)floor((end - h_start[k]) / h_step[k]);
    if (sk < 0) sk = 0;  // CPU loop would run zero times
    h_s[k] = sk;
    total *= sk;
  }

  CUDA_CHECK(cudaMemcpyToSymbol(k_c, h_c, sizeof(h_c)));
  CUDA_CHECK(cudaMemcpyToSymbol(k_d, h_d, sizeof(h_d)));
  CUDA_CHECK(cudaMemcpyToSymbol(k_e, h_e, sizeof(h_e)));
  CUDA_CHECK(cudaMemcpyToSymbol(k_start, h_start, sizeof(h_start)));
  CUDA_CHECK(cudaMemcpyToSymbol(k_step, h_step, sizeof(h_step)));
  CUDA_CHECK(cudaMemcpyToSymbol(k_s, h_s, sizeof(h_s)));

  double* d_out;
  unsigned long long* d_nhits;
  CUDA_CHECK(cudaMalloc(&d_out, CAPACITY * 10 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_nhits, sizeof(unsigned long long)));
  double* h_out = (double*)malloc(CAPACITY * 10 * sizeof(double));
  if (h_out == NULL) {
    printf("Error: host allocation failed\n");
    return 1;
  }

  FILE* fptr = fopen("./results-v1.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  long long pnts = 0;
  long long chunk_max = 1LL << 28;
  long long chunk = chunk_max;

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);

  long long offset = 0;
  while (offset < total) {
    long long cnt = (total - offset < chunk) ? (total - offset) : chunk;
    unsigned long long zero = 0, hits = 0;
    CUDA_CHECK(cudaMemcpy(d_nhits, &zero, sizeof(zero), cudaMemcpyHostToDevice));

    long long blocks = (cnt + BLOCK - 1) / BLOCK;
    gridkernel<<<(unsigned int)blocks, BLOCK>>>(offset, cnt, d_nhits, d_out,
                                                CAPACITY);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&hits, d_nhits, sizeof(hits), cudaMemcpyDeviceToHost));

    if (hits > (unsigned long long)CAPACITY) {
      // output buffer overflowed: redo this range with a smaller chunk
      if (cnt == 1) {
        printf("Error: capacity too small\n");
        return 1;
      }
      chunk = cnt / 2;
      continue;
    }

    CUDA_CHECK(cudaMemcpy(h_out, d_out, hits * 10 * sizeof(double),
                          cudaMemcpyDeviceToHost));
    for (unsigned long long h = 0; h < hits; ++h) {
      for (int k = 0; k < 9; ++k) fprintf(fptr, "%lf\t", h_out[h * 10 + k]);
      fprintf(fptr, "%lf\n", h_out[h * 10 + 9]);
    }
    pnts += (long long)hits;
    offset += cnt;
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);

  fclose(fptr);
  printf("result pnts: %lld\n", pnts);
  printf("Total time = %f seconds\n",
         (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
             (end_main.tv_sec - begin_grid.tv_sec));

  free(h_out);
  cudaFree(d_out);
  cudaFree(d_nhits);
  return EXIT_SUCCESS;
}

