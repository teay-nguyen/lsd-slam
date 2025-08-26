#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel: each thread adds one element
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
  int N = 1 << 16;  // 65536 elements
  size_t size = N * sizeof(float);

  // Allocate host memory
  std::vector<float> h_A(N, 1.0f); // all 1s
  std::vector<float> h_B(N, 2.0f); // all 2s
  std::vector<float> h_C(N);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);

  // Copy input data to device
  cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

  // Launch kernel with enough blocks/threads
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Copy result back to host
  cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);

  // Verify result
  bool success = true;
  for (int i = 0; i < N; i++) {
    if (h_C[i] != 3.0f) {
      success = false;
      break;
    }
  }

  std::cout << (success ? "PASS" : "FAIL") << std::endl;

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}

