#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#define NSEC_SEC_MUL (1.0e9)

// Grid points handled per Thrust call. The index buffer has this many
// entries, so a chunk can never overflow it (every point could be a hit).
#define CHUNK (1LL << 25)

struct timespec begin_grid, end_main;

// to store values of disp.txt: 10 rows of (c_i1..c_i10, d_i, ey_i)
double a[120];

// to store values of grid.txt: 10 rows of (start, end, step)
double b[30];

// Predicate: decodes a linear index into (r1..r10) with r10 as the least
// significant digit, evaluates the 10 constraints, returns true on a hit.
// Problem data is held by value; Thrust passes the functor as a kernel
// argument, so every thread reads it from the parameter bank.
struct InRange {
  double c[10][10];
  double d[10];
  double e[10];
  double start[10];
  double step[10];
  long long s[10];

  __host__ __device__ bool operator()(long long n) const {
    double x[10];
    for (int k = 9; k >= 0; --k) {
      long long r = n % s[k];
      n /= s[k];
      x[k] = start[k] + r * step[k];
    }
    for (int i = 0; i < 10; ++i) {
      double sum = c[i][0] * x[0];
      for (int k = 1; k < 10; ++k) sum += c[i][k] * x[k];
      if (!(fabs(sum - d[i]) <= e[i])) return false;
    }
    return true;
  }
};

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

  InRange pred;
  for (i = 0; i < 10; ++i) {
    for (j = 0; j < 10; ++j) pred.c[i][j] = a[12 * i + j];
    pred.d[i] = a[12 * i + 10];
    pred.e[i] = kk * a[12 * i + 11];
  }

  long long total = 1;
  for (int k = 0; k < 10; ++k) {
    pred.start[k] = b[3 * k];
    double end = b[3 * k + 1];
    pred.step[k] = b[3 * k + 2];
    long long sk = (long long)floor((end - pred.start[k]) / pred.step[k]);
    if (sk < 0) sk = 0;  // CPU loop would run zero times
    pred.s[k] = sk;
    total *= sk;
  }

  FILE* fptr = fopen("./results-v3.txt", "w");
  if (fptr == NULL) {
    printf("Error in creating file !");
    exit(1);
  }

  thrust::device_vector<long long> d_hits(CHUNK);
  thrust::host_vector<long long> h_hits(CHUNK);

  long long pnts = 0;

  clock_gettime(CLOCK_MONOTONIC_RAW, &begin_grid);

  for (long long offset = 0; offset < total; offset += CHUNK) {
    long long cnt = (total - offset < CHUNK) ? (total - offset) : CHUNK;

    // copy_if keeps the indices (in increasing order) whose point passes
    thrust::counting_iterator<long long> first(offset);
    thrust::counting_iterator<long long> last(offset + cnt);
    long long hits =
        thrust::copy_if(thrust::device, first, last, d_hits.begin(), pred) -
        d_hits.begin();

    thrust::copy(d_hits.begin(), d_hits.begin() + hits, h_hits.begin());

    // decode hit indices back to x values (same arithmetic as the CPU)
    for (long long h = 0; h < hits; ++h) {
      long long n = h_hits[h];
      double x[10];
      for (int k = 9; k >= 0; --k) {
        long long r = n % pred.s[k];
        n /= pred.s[k];
        x[k] = pred.start[k] + r * pred.step[k];
      }
      for (int k = 0; k < 9; ++k) fprintf(fptr, "%lf\t", x[k]);
      fprintf(fptr, "%lf\n", x[9]);
    }
    pnts += hits;
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end_main);

  fclose(fptr);
  printf("result pnts: %lld\n", pnts);
  printf("Total time = %f seconds\n",
         (end_main.tv_nsec - begin_grid.tv_nsec) / NSEC_SEC_MUL +
             (end_main.tv_sec - begin_grid.tv_sec));

  return EXIT_SUCCESS;
}

// Compile (set -arch to the target GPU, e.g. sm_80). fmad=false (device) and
// -ffp-contract=off (host decode) keep the arithmetic identical to the CPU:
//   nvcc -O3 -arch=sm_80 --fmad=false -Xcompiler -ffp-contract=off \
//        -o problem2-v3 problem2-v3.cu
