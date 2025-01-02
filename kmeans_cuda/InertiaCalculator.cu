#include "InertiaCalculator.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Simple GPU Error Check Macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
            << " " << file << ":" << line << std::endl;
        exit(code);
    }
}

__global__ void computeInertiaKernel(const float* distances,
    const int* assignments,
    float* partialSums,
    int numPoints,
    int numClusters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        int cluster = assignments[idx];
        float dist = distances[idx * numClusters + cluster];
        atomicAdd(&partialSums[0], dist * dist);
    }
}

float InertiaCalculator::computeInertia(float* d_distances,
    int* d_assignments,
    int numPoints,
    int numClusters)
{
    float h_partialSum = 0.0f;
    float* d_partialSum = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_partialSum, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_partialSum, 0, sizeof(float)));

    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;
    computeInertiaKernel << <gridSize, blockSize >> > (d_distances,
        d_assignments,
        d_partialSum,
        numPoints,
        numClusters);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_partialSum, d_partialSum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partialSum));

    return h_partialSum;
}
