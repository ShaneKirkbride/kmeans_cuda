#include "SilhouetteCalculator.h"
#include <cuda_runtime.h>
#include <cmath>
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

/**
 * Kernel: For demonstration, we approximate:
 *   a = distance to own centroid
 *   b = min distance to other centroids
 */
__global__ void computeABKernel(const float* distances,
    const int* assignments,
    float* aVals,
    float* bVals,
    int numPoints,
    int numClusters)
{
    int pointId = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointId < numPoints) {
        int myCluster = assignments[pointId];
        float distToMyCluster = distances[pointId * numClusters + myCluster];

        // "a" = distance to assigned centroid (approx)
        float a = distToMyCluster;

        // "b" = min distance to other centroids
        float b = 1e9f;
        for (int c = 0; c < numClusters; c++) {
            if (c == myCluster) continue;
            float dist = distances[pointId * numClusters + c];
            if (dist < b) {
                b = dist;
            }
        }

        aVals[pointId] = a;
        bVals[pointId] = b;
    }
}

float SilhouetteCalculator::computeSilhouette(float* d_distances,
    int* d_assignments,
    int numPoints,
    int numClusters)
{
    // Allocate device memory for a and b
    float* d_aVals = nullptr;
    float* d_bVals = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_aVals, sizeof(float) * numPoints));
    CUDA_CHECK(cudaMalloc((void**)&d_bVals, sizeof(float) * numPoints));

    int blockSize = 256;
    int gridSize = (numPoints + blockSize - 1) / blockSize;

    // Compute a and b for each point
    computeABKernel <<<gridSize, blockSize >>> (d_distances,
        d_assignments,
        d_aVals,
        d_bVals,
        numPoints,
        numClusters);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    std::vector<float> h_aVals(numPoints), h_bVals(numPoints);
    CUDA_CHECK(cudaMemcpy(h_aVals.data(), d_aVals, sizeof(float) * numPoints, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bVals.data(), d_bVals, sizeof(float) * numPoints, cudaMemcpyDeviceToHost));

    // Compute silhouette on host
    float sumSil = 0.0f;
    for (int i = 0; i < numPoints; ++i) {
        float a = h_aVals[i];
        float b = h_bVals[i];
        float s = (b - a) / fmaxf(a, b);
        sumSil += s;
    }
    float meanSilhouette = sumSil / numPoints;

    CUDA_CHECK(cudaFree(d_aVals));
    CUDA_CHECK(cudaFree(d_bVals));

    return meanSilhouette;
}
