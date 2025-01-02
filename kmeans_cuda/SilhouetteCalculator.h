#pragma once

/**
 * @class SilhouetteCalculator
 * @brief Computes a simplified silhouette score for each point based on centroid distances.
 *
 * Single Responsibility:
 * - Calculates silhouette for each data point and aggregates a mean silhouette score.
 */
class SilhouetteCalculator {
public:
    /**
     * @brief Computes the average silhouette score across all data points.
     * @param d_distances Pointer to distance array on device
     * @param d_assignments Pointer to cluster assignments on device
     * @param numPoints Number of data points
     * @param numClusters Number of clusters
     * @return The mean silhouette score
     */
    float computeSilhouette(float* d_distances,
        int* d_assignments,
        int numPoints,
        int numClusters);
};
