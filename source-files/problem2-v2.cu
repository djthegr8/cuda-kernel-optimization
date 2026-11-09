#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define NSEC_SEC_MUL (1.0e9)
#ifndef BLOCK
#define BLOCK 128
#endif
#ifndef PRUNE
#define PRUNE 0  // ablation switch: 1 enables pruning (kept off by default)
#endif
#define NPREFIX 7             // variables x1..x7 are fixed per thread
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

__constant__ double k_c[10][10];
__constant__ double k_d[10];
__constant__ double k_e[10];
__constant__ double k_start[10];
__constant__ double k_step[10];
__constant__ long long k_s[10];

// Append one result row. Threads that reach this point together (the
// coalesced group) reserve their slots with a single atomicAdd.
__device__ __forceinline__ void record(const double* x,
                                       unsigned long long* nhits, double* out,
                                       long long capacity) {
  cg::coalesced_group g = cg::coalesced_threads();
  unsigned long long base = 0;
  if (g.thread_rank() == 0)
    base = atomicAdd(nhits, (unsigned long long)g.size());
  base = g.shfl(base, 0);
  unsigned long long slot = base + g.thread_rank();
  if (slot < (unsigned long long)capacity) {
    for (int k = 0; k < 10; ++k) out[slot * 10 + k] = x[k];
  }
}

// One thread = one prefix (r1..r7). It computes the 10 partial sums over
// x1..x7 once, then loops over r8, r9, r10 adding one term per level.
// Summation order is left to right, same as the CPU expression.
//
// prune != 0 requires c >= 0, start >= 0, step > 0. Then every added term is
// >= 0, partial sums never decrease, and x grows with r, so once
// (partial - d > e) holds for some constraint, it holds for every later
// term at that level and below. The loop at that level can stop.
__global__ void gridkernel(long long offset, long long count, int prune,
                           unsigned long long* nhits, double* out,
                           long long capacity) {
  long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count) return;
  long long n = offset + t;

  double x[10];
  for (int k = NPREFIX - 1; k >= 0; --k) {
    long long r = n % k_s[k];
    n /= k_s[k];
    x[k] = k_start[k] + r * k_step[k];
  }

  double p7[10];
  bool over = false;
  for (int i = 0; i < 10; ++i) {
    double sum = k_c[i][0] * x[0];
    for (int k = 1; k < NPREFIX; ++k) sum += k_c[i][k] * x[k];
    p7[i] = sum;
    if (prune && (sum - k_d[i] > k_e[i])) over = true;
  }
  if (over) return;

  for (long long r8 = 0; r8 < k_s[7]; ++r8) {
    x[7] = k_start[7] + r8 * k_step[7];
    double p8[10];
    over = false;
    for (int i = 0; i < 10; ++i) {
      p8[i] = p7[i] + k_c[i][7] * x[7];
      if (prune && (p8[i] - k_d[i] > k_e[i])) over = true;
    }
    if (over) break;

    for (long long r9 = 0; r9 < k_s[8]; ++r9) {
      x[8] = k_start[8] + r9 * k_step[8];
      double p9[10];
      over = false;
      for (int i = 0; i < 10; ++i) {
        p9[i] = p8[i] + k_c[i][8] * x[8];
        if (prune && (p9[i] - k_d[i] > k_e[i])) over = true;
      }
      if (over) break;

      for (long long r10 = 0; r10 < k_s[9]; ++r10) {
        x[9] = k_start[9] + r10 * k_step[9];
        bool ok = true;
        over = false;
        for (int i = 0; i < 10; ++i) {
          double diff = (p9[i] + k_c[i][9] * x[9]) - k_d[i];
          if (!(fabs(diff) <= k_e[i])) {
            // first failing constraint: skip the rest, like the CPU's early exit
            ok = false;
            over = prune && (diff > k_e[i]);
            break;
          }
        }
        if (over) break;
        if (ok) record(x, nhits, out, capacity);
      }
    }
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

  double h_c[10][10], h_d[10], h_e[10];
  for (i = 0; i < 10; ++i) {
    for (j = 0; j < 10; ++j) h_c[i][j] = a[12 * i + j];
    h_d[i] = a[12 * i + 10];
    h_e[i] = kk * a[12 * i + 11];
  }

  double h_start[10], h_step[10];
  long long h_s[10];
  long long total_prefix = 1;  // number of (r1..r7) combinations
  for (int k = 0; k < 10; ++k) {
    h_start[k] = b[3 * k];
    double end = b[3 * k + 1];
    h_step[k] = b[3 * k + 2];
    long long sk = (long long)floor((end - h_start[k]) / h_step[k]);
    if (sk < 0) sk = 0;  // CPU loop would run zero times
    h_s[k] = sk;
    if (k < NPREFIX) total_prefix *= sk;
  }

  // pruning is valid only for nonnegative coefficients and x values
  int prune = PRUNE;
  for (i = 0; i < 10; ++i)
    for (j = 0; j < 10; ++j)
      if (!(h_c[i][j] >= 0.0)) prune = 0;
  for (int k = 0; k < 10; ++k)
    if (!(h_start[k] >= 0.0) || !(h_step[k] > 0.0)) prune = 0;
  fprintf(stderr, "pruning: %s\n", prune ? "on" : "off");

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

  FILE* fptr = fopen("./results-v2.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  long long pnts = 0;
  long long chunk = 1LL << 22;  // prefixes per launch

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);

  long long offset = 0;
  while (offset < total_prefix) {
    long long cnt =
        (total_prefix - offset < chunk) ? (total_prefix - offset) : chunk;
    unsigned long long zero = 0, hits = 0;
    CUDA_CHECK(cudaMemcpy(d_nhits, &zero, sizeof(zero), cudaMemcpyHostToDevice));

    long long blocks = (cnt + BLOCK - 1) / BLOCK;
    gridkernel<<<(unsigned int)blocks, BLOCK>>>(offset, cnt, prune, d_nhits,
                                                d_out, CAPACITY);
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

