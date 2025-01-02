#include "KMeans.h"
#include "Dataset.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

// Simple GPU Error Check Macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
            << " " << file << ":" << line << std::endl;
        exit(code);
    }
}

// Kernel to assign clusters based on precomputed distances
__global__ void assignClustersKernel(const float* distances,
    int* assignments,
    int numPoints,
    int numClusters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        float bestDist = 1e9f;
        int bestClusterId = -1;
        for (int c = 0; c < numClusters; c++) {
            float d = distances[idx * numClusters + c];
            if (d < bestDist) {
                bestDist = d;
                bestClusterId = c;
            }
        }
        assignments[idx] = bestClusterId;
    }
}

// Kernel to accumulate partial sums for centroid updates
__global__ void updateCentroidsKernel(const float* data,
    const int* assignments,
    float* centroids,
    float* centroidCounts,
    int numPoints,
    int numClusters,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        int clusterId = assignments[idx];
        for (int d = 0; d < dim; d++) {
            atomicAdd(&centroids[clusterId * dim + d], data[idx * dim + d]);
        }
        atomicAdd(&centroidCounts[clusterId], 1.0f);
    }
}

// Constructor
KMeans::KMeans(std::unique_ptr<IDistanceCalculator> distCalc)
    : distanceCalculator_(std::move(distCalc))
{
}

std::tuple<int, float, float> KMeans::findOptimalClusters(Dataset& dataset,
    int maxK,
    int maxIterations)
{
    int bestK = 2;
    float bestSilhouette = -1.0f;
    float bestInertia = 1e9f;

    for (int k = 2; k <= maxK; ++k) {
        auto result = runKMeans(dataset, k, maxIterations);
        float inertia = std::get<0>(result);
        float sil = std::get<1>(result);
        // Simple heuristic: choose the cluster # with best silhouette
        if (sil > bestSilhouette) {
            bestSilhouette = sil;
            bestK = k;
            bestInertia = inertia;
        }
    }

    return { bestK, bestInertia, bestSilhouette };
}

std::pair<float, float> KMeans::runKMeans(Dataset& dataset, int k, int maxIterations)
{
    int numPoints = dataset.getNumPoints();
    int dim = dataset.getDim();

    // Host buffer for initial centroids
    std::vector<float> h_centroids(k * dim);

    // Simple init: each cluster c, dimension d => 0.01f * ((c+1)*(d+1))
    for (int c = 0; c < k; ++c) {
        for (int d = 0; d < dim; d++) {
            h_centroids[c * dim + d] = 0.01f * float((c + 1) * (d + 1));
        }
    }

    // Allocate device memory
    float* d_centroids = nullptr;
    float* d_distances = nullptr;
    int* d_assignments = nullptr;
    float* d_centroidCounts = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_centroids, sizeof(float) * k * dim));
    CUDA_CHECK(cudaMalloc((void**)&d_distances, sizeof(float) * numPoints * k));
    CUDA_CHECK(cudaMalloc((void**)&d_assignments, sizeof(int) * numPoints));
    CUDA_CHECK(cudaMalloc((void**)&d_centroidCounts, sizeof(float) * k));

    // Copy initial centroids
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(),
        sizeof(float) * k * dim, cudaMemcpyHostToDevice));

    // Main KMeans loop
    for (int iter = 0; iter < maxIterations; ++iter) {
        // 1. Compute distances
        distanceCalculator_->computeDistances(dataset.getDeviceData(),
            d_centroids,
            d_distances,
            numPoints,
            k,
            dim);

        // 2. Assign clusters
        {
            int blockSize = 256;
            int gridSize = (numPoints + blockSize - 1) / blockSize;
            assignClustersKernel << <gridSize, blockSize >> > (d_distances,
                d_assignments,
                numPoints,
                k);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 3. Reset centroids and counts
        CUDA_CHECK(cudaMemset(d_centroids, 0, sizeof(float) * k * dim));
        CUDA_CHECK(cudaMemset(d_centroidCounts, 0, sizeof(float) * k));

        // 4. Accumulate sums for new centroids
        {
            int blockSize = 256;
            int gridSize = (numPoints + blockSize - 1) / blockSize;
            updateCentroidsKernel<<<gridSize, blockSize>>> (
                dataset.getDeviceData(),
                d_assignments,
                d_centroids,
                d_centroidCounts,
                numPoints,
                k,
                dim
                );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 5. Compute mean for each centroid on host (demo approach)
        {
            // Copy partial sums back
            std::vector<float> h_centroidCounts(k);
            std::vector<float> h_centroidsTemp(k * dim);

            CUDA_CHECK(cudaMemcpy(h_centroidCounts.data(), d_centroidCounts,
                sizeof(float) * k, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_centroidsTemp.data(), d_centroids,
                sizeof(float) * k * dim, cudaMemcpyDeviceToHost));

            // Compute mean
            for (int c = 0; c < k; ++c) {
                float count = (h_centroidCounts[c] < 1e-9f) ? 1.0f : h_centroidCounts[c];
                for (int d = 0; d < dim; d++) {
                    h_centroidsTemp[c * dim + d] /= count;
                }
            }

            // Copy updated centroids back
            CUDA_CHECK(cudaMemcpy(d_centroids, h_centroidsTemp.data(),
                sizeof(float) * k * dim, cudaMemcpyHostToDevice));
        }
    }

    // Compute final distances
    distanceCalculator_->computeDistances(dataset.getDeviceData(),
        d_centroids,
        d_distances,
        numPoints,
        k,
        dim);

    // Compute inertia
    float inertia = inertiaCalculator_.computeInertia(d_distances,
        d_assignments,
        numPoints,
        k);

    // Compute silhouette
    float silhouette = silhouetteCalculator_.computeSilhouette(d_distances,
        d_assignments,
        numPoints,
        k);

    // Free device memory
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_centroidCounts));

    return { inertia, silhouette };
}
