#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>

#define THRESHOLD (std::numeric_limits<float>::epsilon())

using std::cout;
using std::endl;

using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define cudaCheckError(ans)                                                    \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

// TODO: Evaluate with three sizes, 64, 128, and 256.
// Changed from uint -> int to prevent issues when using -Filter_Width
const int32_t Input_Width = (1 << 8);
const int32_t Input_Height = (1 << 8);
const int32_t Input_Num_Elems = (Input_Width * Input_Height * Input_Height);
const int32_t Input_Size_Bytes = Input_Num_Elems * sizeof(float);

const int32_t Filter_Width = 5;
const int32_t Filter_Num_Elems = (Filter_Width * Filter_Width * Filter_Width);
const int32_t Filter_Size_Bytes = Filter_Num_Elems * sizeof(float);

__constant__ float d_filter_cm[Filter_Num_Elems];
const int32_t Tile_Width = 8;
const int32_t Num_Threads = Tile_Width*Tile_Width*Tile_Width; // 512
const int32_t Shared_Width = 12;
const int32_t Total_Shared_Elems = Shared_Width*Shared_Width*Shared_Width; // 1728
const int32_t Radius = 2; // (Shared_Width - Tile_Width)/2;
// TODO: Implement a single-threaded CPU version
void seq_cpu_conv(const float *h_input, const float *h_filter,
                  float *h_output){
  for (int i = 0; i < Input_Width; i++){
    for (int j = 0; j < Input_Width; j++){
      for (int k = 0; k < Input_Width; k++){
        float sum = 0.0f;
        for (int x = -Filter_Width/2; x <= Filter_Width/2; x++){
          for (int y = -Filter_Width/2; y <= Filter_Width/2; y++){
            for (int z = -Filter_Width/2; z <= Filter_Width/2; z++){
              if (i+x >= 0 && i+x < Input_Width && j+y >= 0 && j+y < Input_Width && k+z >= 0 && k + z < Input_Width)
              sum += h_filter[(x + Filter_Width/2)*Filter_Width*Filter_Width +(y + Filter_Width/2)*Filter_Width + z + Filter_Width/2]*h_input[(i+x)*Input_Width*Input_Width + (j+y)*Input_Width + k+z];
            }
          }
        }
        h_output[i*Input_Width*Input_Width + j*Input_Width + k] = sum;
      }
    }
  }
}

// TODO: Implement a naive version using loads and stores to global memory. Edit
// the function definition as required.
__global__ void basic_kernel(const float* d_input, const float* d_filter, float* d_output) {
  int i = blockIdx.x*blockDim.x + threadIdx.x, j = blockIdx.y*blockDim.y + threadIdx.y, k = blockIdx.z*blockDim.z + threadIdx.z;
  float sum = 0.0f;
  for (int x = -Filter_Width/2; x <= Filter_Width/2; x++){
    for (int y = -Filter_Width/2; y <= Filter_Width/2; y++){
      for (int z = -Filter_Width/2; z <= Filter_Width/2; z++){
        if (i+x >= 0 && i+x < Input_Width && j+y >= 0 && j+y < Input_Width && k+z >= 0 && k + z < Input_Width)
          sum += d_filter[(x + Filter_Width/2)*Filter_Width*Filter_Width +(y + Filter_Width/2)*Filter_Width + z + Filter_Width/2]*d_input[(i+x)*Input_Width*Input_Width + (j+y)*Input_Width + k+z];
      }
    }
  }
  d_output[i*Input_Width*Input_Width + j*Input_Width + k] = sum;
}

// TODO: Use constant memory for accessing the filter and shared memory for
// storing the input array including halo cells.
__global__ void opt_kernel(const float* d_input, float* d_output) {
  __shared__ float s_input[12][12][12];
  int tid = threadIdx.x * (Tile_Width * Tile_Width) + threadIdx.y * Tile_Width + threadIdx.z;
  for (int idx = tid; idx < Total_Shared_Elems; idx += Num_Threads){
    int sx = idx/(Shared_Width*Shared_Width);
    int sy = (idx/Shared_Width) % Shared_Width;
    int sz = idx % Shared_Width;
    int gx = blockIdx.x*Tile_Width + sx - Radius;
    int gy = blockIdx.y*Tile_Width + sy - Radius;
    int gz = blockIdx.z*Tile_Width + sz - Radius;
    if (gx >= 0 && gx < Input_Width &&
        gy >= 0 && gy < Input_Width &&
        gz >= 0 && gz < Input_Width) {
      s_input[sx][sy][sz] = d_input[gx * Input_Width * Input_Width + gy * Input_Width + gz];
    } else {
      s_input[sx][sy][sz] = 0.0f; // Out-of-bounds halo cell = 0
    }
  }

  __syncthreads();
  int i = blockIdx.x * Tile_Width + threadIdx.x;
  int j = blockIdx.y * Tile_Width + threadIdx.y;
  int k = blockIdx.z * Tile_Width + threadIdx.z;

  float sum = 0.0f;
  #pragma unroll
  for (int x = -Radius; x <= Radius; x++){
    #pragma unroll
    for (int y = -Radius; y <= Radius; y++){
      #pragma unroll
      for (int z = -Radius; z <= Radius; z++){
          float f_val = d_filter_cm[(x + Radius) * Filter_Width * Filter_Width +
                                 (y + Radius) * Filter_Width +
                                 (z + Radius)];
          float i_val = s_input[threadIdx.x + Radius + x][threadIdx.y + Radius + y][threadIdx.z + Radius + z];
          sum += f_val * i_val;      
        }
    }
  }
  d_output[i*Input_Width*Input_Width + j*Input_Width + k] = sum;

}

void check_result(const float *w_ref, const float *w_opt, const uint32_t N) {
  double maxdiff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < (N * N * N); i++) {
    // modified to do fabs here to prevent getting maxdiff 0.
    double this_diff = fabs(w_ref[i] - w_opt[i]);
    if (this_diff > THRESHOLD) {
      numdiffs++;
      if (this_diff > maxdiff) {
        maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

void print3D(const float *A, const uint32_t N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        cout << A[i * N * N + j * N + k] << "\t";
      }
      cout << "\n";
    }
    cout << "\n";
  }
}

int main() {
  auto *h_input = new float[Input_Num_Elems];
  std::fill_n(h_input, Input_Num_Elems, 1.0F);

  const float h_filter3D[Filter_Width][Filter_Width][Filter_Width] = {
      {2.0F, 3.0F, 4.0F, 3.0F, 2.0F},
      {4.0F, 5.0F, 6.0F, 5.0F, 4.0F},
      {3.0F, 4.0F, 5.0F, 4.0F, 3.0F}};
  auto *h_filter = new float[Filter_Num_Elems];
  memcpy(h_filter, h_filter3D, Filter_Size_Bytes);

  auto *h_output_cpu = new float[Input_Num_Elems];
  HRTimer start = HR::now();
  seq_cpu_conv(h_input, h_filter, h_output_cpu);
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Serial version: " << duration << " ms\n";

  float *d_input = nullptr;
  float *d_filter = nullptr;
  float *d_output = nullptr;
  auto* h_output_basic = new float[Input_Num_Elems];
  auto* h_output_opt = new float[Input_Num_Elems];
  float kernel_time = 0.0f, endToEnd_time = 0.0f;

  // TODO: Invoke kernel3D_basic()
  // TODO: Adapt check_result()
  cudaEvent_t estt, estp, stt, stp;
  cudaEventCreate(&stt);
  cudaEventCreate(&stp);
  cudaEventCreate(&estt);
  cudaEventCreate(&estp);

  // ------ BASIC KERNEL ------
  cudaMalloc(&d_input, Input_Size_Bytes);
  cudaMalloc(&d_filter, Filter_Size_Bytes);
  cudaMalloc(&d_output, Input_Size_Bytes);
  cudaEventRecord(estt);

  cudaMemcpy(d_input, h_input, Input_Size_Bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_filter, h_filter, Filter_Size_Bytes, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock (8,8,8);
  dim3 blocksInGrid(Input_Width/8, Input_Width/8, Input_Width/8);
  cudaEventRecord(stt);
  basic_kernel<<<blocksInGrid, threadsPerBlock>>>(d_input, d_filter, d_output);
  cudaEventRecord(stp);
  cudaMemcpy(h_output_basic, d_output, Input_Size_Bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(estp);
  cudaEventSynchronize(estp);
  cudaEventElapsedTime(&kernel_time, stt, stp);
  cudaEventElapsedTime(&endToEnd_time, estt, estp);
  cout << "- - - - - - - Basic Kernel - - - - - - - -" << endl;
  check_result(h_output_basic, h_output_cpu, Input_Width);
  cout << "Kernel3D_basic time (ms): " << kernel_time << "\n";
  cout << "Kernel3D_basic End-To-End time (ms): " << endToEnd_time << "\n";

  // ------ OPTIMAL KERNEL ------
  /*
  I've redone the memcpy for the comparison of end to end times to be valid.
  */
  cudaMemset(d_output, 0, Input_Size_Bytes);
  cudaEventRecord(estt);
  cudaMemcpyToSymbol(d_filter_cm, h_filter, Filter_Size_Bytes);
  cudaMemcpy(d_input, h_input, Input_Size_Bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(stt);
  opt_kernel<<<blocksInGrid, threadsPerBlock>>>(d_input, d_output);
  cudaEventRecord(stp);
  cudaMemcpy(h_output_opt, d_output, Input_Size_Bytes, cudaMemcpyDeviceToHost);
  cudaEventRecord(estp);
  cudaEventSynchronize(estp);
  cudaEventElapsedTime(&kernel_time, stt, stp);
  cudaEventElapsedTime(&endToEnd_time, estt, estp);
  cout << "- - - - - - Optimal Kernel - - - - - - -" << endl;
  check_result(h_output_opt, h_output_cpu, Input_Width);
  cout << "Kernel3D_opt time (ms): " << kernel_time << "\n";
  cout << "Kernel3D_opt End-To-End Time (ms): " << endToEnd_time << "\n";

  // TODO: Free memory
  cudaEventDestroy(stp);
  cudaEventDestroy(stt);
  cudaEventDestroy(estp);
  cudaEventDestroy(estt);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_filter);
  delete[] h_input;
  delete[] h_filter;
  delete[] h_output_cpu;
  delete[] h_output_basic;
  delete[] h_output_opt;
  return EXIT_SUCCESS;
}