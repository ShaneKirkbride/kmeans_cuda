#pragma once

/**
 * @class InertiaCalculator
 * @brief Responsible for computing KMeans inertia on the GPU.
 *
 * Single Responsibility:
 * - Sums up squared distances of points to their assigned centroids.
 */
class InertiaCalculator {
public:
    /**
     * @brief Computes the sum of squared distances (inertia).
     * @param d_distances Pointer to distance array on device
     * @param d_assignments Pointer to cluster assignments on device
     * @param numPoints Number of data points
     * @param numClusters Number of clusters
     * @return The computed inertia (host float)
     */
    float computeInertia(float* d_distances,
        int* d_assignments,
        int numPoints,
        int numClusters);
};