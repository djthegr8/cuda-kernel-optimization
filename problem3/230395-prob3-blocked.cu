#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
using namespace std;

#define BLOCK 256
#define MAX_GRID 65535
#ifndef BLOCK_COUNTERS
#define BLOCK_COUNTERS 32u
#endif
static_assert(BLOCK_COUNTERS % 4 == 0, "a block must hold whole 32-bit words");

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),  \
              __FILE__, __LINE__);                                           \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// Hash functions generated from claude
// hash1: Murmur3 finalizer (xor-shift and multiply)
__device__ uint32_t hash1(uint32_t key) {
  uint32_t h = key;
  h ^= h >> 16;  h *= 0x85ebca6bu;
  h ^= h >> 13;  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h % BLOCK_COUNTERS;
}

// hash2: Knuth multiplicative hash, range reduction without a division
__device__ uint32_t hash2(uint32_t key) {
  uint32_t h = key * 2654435769u;
  return (uint32_t)(((uint64_t)h * BLOCK_COUNTERS) >> 32);
}

// hash3: "lowbias32" mixer (different constants and shifts from hash1)
__device__ uint32_t hash3(uint32_t key) {
  uint32_t h = key;
  h ^= h >> 16;  h *= 0x7feb352du;
  h ^= h >> 15;  h *= 0x846ca68bu;
  h ^= h >> 16;
  return h % BLOCK_COUNTERS;
}

// hash 4 for blocked
__device__ uint32_t hash4(uint32_t key, uint32_t num_blocks) {
  uint32_t h = key ^ 0x9e3779b9u;
  h ^= h >> 15;  h *= 0x2c1b3c6du;
  h ^= h >> 12;  h *= 0x297a2d39u;
  h ^= h >> 15;
  return (uint32_t)(((uint64_t)h * num_blocks) >> 32);   
}


// A key's three counters all lie in one block: hash4 picks the block, and
// hash1..3 pick offsets inside it. h[] receives absolute counter indices
// (block * BLOCK_COUNTERS + offset), so the word/byte arithmetic in the kernels
// and the key_present() multiplicity check work unchanged.
__device__ __forceinline__ void blocked_indices(uint32_t key, uint32_t num_blocks,
                                                uint32_t h[3]) {
  uint32_t base = hash4(key, num_blocks) * BLOCK_COUNTERS;
  h[0] = base + hash1(key);
  h[1] = base + hash2(key);
  h[2] = base + hash3(key);
}

__global__ void bloom_insert(uint32_t* counter_words, const uint32_t* ops,
                             const size_t n_it, const uint32_t num_blocks) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_it;
       i += stride) {
    uint32_t h[3];
    blocked_indices(ops[i], num_blocks, h);
    for (int t = 0; t < 3; t++) {
      uint32_t el = h[t];
      uint32_t w = el >> 2;
      uint32_t offset = (el & 3) * 8;
      uint32_t mask = 0xFFu << offset;
      uint32_t ctr = counter_words[w];
      while (true) {
        uint32_t ccnt = (((ctr & mask) >> offset) + 1) & 0xFFu;  
        uint32_t seen = ctr;
        ctr = atomicCAS(counter_words + w, seen,
                        (seen & ~mask) | (ccnt << offset));
        if (ctr == seen) break;
      }
    }
  }
}


__device__ __forceinline__ bool key_present(const uint32_t* counter_words,
                                            const uint32_t h[3]) {
  uint32_t minvals[3] = {(uint32_t)((h[0] == h[1]) + (h[0] == h[2])),
                         (uint32_t)((h[1] == h[0]) + (h[1] == h[2])),
                         (uint32_t)((h[2] == h[0]) + (h[2] == h[1]))};
  for (int t = 0; t < 3; t++) {
    uint32_t el = h[t];
    uint32_t ctr = counter_words[el >> 2];
    uint32_t offset = (el & 3) * 8;
    uint32_t ccnt = (ctr >> offset) & 0xFFu;
    if (ccnt <= minvals[t]) return false;
  }
  return true;
}

__global__ void bloom_delete(uint32_t* counter_words, const uint32_t* ops,
                             const size_t n_dt, const uint32_t num_blocks) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_dt;
       i += stride) {
    uint32_t h[3];
    blocked_indices(ops[i], num_blocks, h);
    if (!key_present(counter_words, h)) continue;  

    for (int t = 0; t < 3; t++) {
      uint32_t el = h[t];
      uint32_t w = el >> 2;
      uint32_t offset = (el & 3) * 8;
      uint32_t mask = 0xFFu << offset;
      uint32_t ctr = counter_words[w];
      while (true) {
        uint32_t ccnt = (ctr & mask) >> offset;
        if (ccnt == 0) break;  // never decrement below zero (would smear the word)
        uint32_t seen = ctr;
        ctr = atomicCAS(counter_words + w, seen,
                        (seen & ~mask) | ((ccnt - 1) << offset));
        if (ctr == seen) break;
      }
    }
  }
}

__global__ void bloom_search(const uint32_t* counter_words, const uint32_t* ops,
                             uint8_t* out, const size_t n_st, const uint32_t num_blocks) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_st;
       i += stride) {
    uint32_t h[3];
    blocked_indices(ops[i], num_blocks, h);
    out[i] = key_present(counter_words, h) ? 1 : 0;
  }
}


static uint32_t* d_counters = nullptr;  // counter words on the device
static uint32_t g_num_blocks = 0;       // number of BLOCK_COUNTERS-sized blocks


static bool getArray(const char* path, vector<uint32_t>& keys) {
  keys.clear();
  FILE* f = fopen(path, "rb");
  if (!f) {
    perror("fopen");
    return false;
  }
  fseek(f, 0, SEEK_END);
  long bytes = ftell(f);
  rewind(f);
  if (bytes < 0) {
    fclose(f);
    return false;
  }
  size_t n = (size_t)bytes / sizeof(uint32_t);
  keys.resize(n);
  if (n > 0 && fread(keys.data(), sizeof(uint32_t), n, f) != n) {
    fprintf(stderr, "read error in %s\n", path);
    fclose(f);
    keys.clear();
    return false;
  }
  fclose(f);
  return true;
}

static unsigned grid_for(size_t n) {
  size_t blocks = (n + BLOCK - 1) / BLOCK;
  return (unsigned)(blocks < MAX_GRID ? blocks : MAX_GRID);
}

// Copies keys to the device, runs the kernel, prints the kernel time.
static void insert_host(const vector<uint32_t>& keys) {
  size_t n = keys.size();
  if (n == 0) return;
  uint32_t* d_keys;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s);
  bloom_insert<<<grid_for(n), BLOCK>>>(d_counters, d_keys, n, g_num_blocks);
  cudaEventRecord(e);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventSynchronize(e));
  float ms;
  cudaEventElapsedTime(&ms, s, e);
  printf("insert: %zu keys, kernel time %.3f ms\n", n, ms);
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  CUDA_CHECK(cudaFree(d_keys));
}

static void delete_host(const vector<uint32_t>& keys) {
  size_t n = keys.size();
  if (n == 0) return;
  uint32_t* d_keys;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s);
  bloom_delete<<<grid_for(n), BLOCK>>>(d_counters, d_keys, n, g_num_blocks);
  cudaEventRecord(e);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventSynchronize(e));
  float ms;
  cudaEventElapsedTime(&ms, s, e);
  printf("delete: %zu keys, kernel time %.3f ms\n", n, ms);
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  CUDA_CHECK(cudaFree(d_keys));
}

// Returns one byte per key: 1 = reported present, 0 = absent (chose this because assignment didnt give format)
static vector<uint8_t> search_host(const vector<uint32_t>& keys) {
  size_t n = keys.size();
  vector<uint8_t> out(n, 0);
  if (n == 0) return out;
  uint32_t* d_keys;
  uint8_t* d_out;
  CUDA_CHECK(cudaMalloc(&d_keys, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(uint8_t)));
  CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), n * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  cudaEventRecord(s);
  bloom_search<<<grid_for(n), BLOCK>>>(d_counters, d_keys, d_out, n, g_num_blocks);
  cudaEventRecord(e);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventSynchronize(e));
  float ms;
  cudaEventElapsedTime(&ms, s, e);
  printf("search: %zu keys, kernel time %.3f ms\n", n, ms);
  cudaEventDestroy(s);
  cudaEventDestroy(e);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, n * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_out));
  return out;
}

// Compares the filter's answers with an exact set that mirrors the same
// operations. A false negative (key in set, filter says absent) must not
// happen; a false positive (key not in set, filter says present) is expected.
static void check_result(const char* phase, const vector<uint32_t>& search_keys,
                         const vector<uint8_t>& out,
                         const unordered_set<uint32_t>& truth) {
  size_t fp = 0, fn = 0, neg = 0;
  for (size_t i = 0; i < search_keys.size(); i++) {
    bool in_set = truth.count(search_keys[i]) != 0;
    bool reported = out[i] != 0;
    if (!in_set) neg++;
    if (in_set && !reported) fn++;
    if (!in_set && reported) fp++;
  }
  size_t total = search_keys.size();
  printf("[%s] searches: %zu, false negatives: %zu, false positives: %zu\n",
         phase, total, fn, fp);
  if (total > 0)
    printf("[%s] false positive rate (false positives / all searches): %.6f"
           "   (over absent keys only: %.6f)\n",
           phase, (double)fp / total, neg ? (double)fp / neg : 0.0);
  if (fn != 0) printf("[%s] WARNING: false negatives found!\n", phase); // This was happening for very small N due to overflow
}

static void dedup(vector<uint32_t>& v) {
  sort(v.begin(), v.end());
  v.erase(unique(v.begin(), v.end()), v.end());
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    cout << "Wrong usage!\nArguments: N <abs-path-insert-trace> "
            "<abs-path-search-trace> <abs-path-delete-trace>"
         << endl;
    return 1;
  }
  uint32_t N = (uint32_t)strtoul(argv[1], NULL, 0);
  if (N == 0) {
    cout << "Invalid N" << endl;
    return 1;
  }
  vector<uint32_t> insert_keys, search_keys, delete_keys;
  if (!getArray(argv[2], insert_keys) || !getArray(argv[3], search_keys) ||
      !getArray(argv[4], delete_keys)) {
    cout << "Exiting due to read errors" << endl;
    return 1;
  }

  // duplicates in the insert and delete traces are removed
  dedup(insert_keys);
  dedup(delete_keys);

  // N counters are grouped into blocks of BLOCK_COUNTERS; a trailing partial
  // block is not used, so the counter array has num_blocks * BLOCK_COUNTERS entries.
  if (N < BLOCK_COUNTERS) {
    cout << "N must be at least " << BLOCK_COUNTERS << endl;
    return 1;
  }
  g_num_blocks = N / BLOCK_COUNTERS;
  uint32_t N_used = g_num_blocks * BLOCK_COUNTERS;
  printf("N = %u requested, %u counters used (%u blocks of %u)\n", N, N_used,
         g_num_blocks, BLOCK_COUNTERS);
  // 4 byte words, so 1 word has 4 counters.
  size_t words = (size_t)N_used / 4;
  CUDA_CHECK(cudaMalloc(&d_counters, words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_counters, 0, words * sizeof(uint32_t)));

  unordered_set<uint32_t> truth;  // gt of filter

  insert_host(insert_keys);
  for (uint32_t k : insert_keys) truth.insert(k);
  check_result("after insert", search_keys, search_host(search_keys), truth);

  delete_host(delete_keys);
  for (uint32_t k : delete_keys) truth.erase(k);
  check_result("after delete", search_keys, search_host(search_keys), truth);

  CUDA_CHECK(cudaFree(d_counters));
  return 0;
}
