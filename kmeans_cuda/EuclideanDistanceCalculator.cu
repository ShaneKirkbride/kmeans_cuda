#include "EuclideanDistanceCalculator.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// Simple GPU Error Check Macro (could share via a common header if desired)
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
            << " " << file << ":" << line << std::endl;
        exit(code);
    }
}

__global__ void computeEuclideanDistancesKernel(const float* data,
    const float* centroids,
    float* outDistances,
    int numPoints,
    int numClusters,
    int dim)
{
    // 2D grid: each thread is (pointId, clusterId)
    int pointId = blockIdx.x * blockDim.x + threadIdx.x;
    int clusterId = blockIdx.y * blockDim.y + threadIdx.y;

    if (pointId < numPoints && clusterId < numClusters) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = data[pointId * dim + d] - centroids[clusterId * dim + d];
            sum += diff * diff;
        }
        outDistances[pointId * numClusters + clusterId] = sqrtf(sum);
    }
}

void EuclideanDistanceCalculator::computeDistances(float* d_data,
    float* d_centroids,
    float* d_outDistances,
    int numPoints,
    int numClusters,
    int dim)
{
    dim3 block(16, 16);
    dim3 grid((numPoints + block.x - 1) / block.x,
        (numClusters + block.y - 1) / block.y);

    computeEuclideanDistancesKernel << <grid, block >> > (d_data,
        d_centroids,
        d_outDistances,
        numPoints,
        numClusters,
        dim);

    CUDA_CHECK(cudaDeviceSynchronize());
}
