#pragma once

/**
 * @interface IDistanceCalculator
 * @brief Abstract interface for distance calculations (e.g., Euclidean).
 *
 * Single Responsibility:
 * - Defines the contract for computing distances between data points and centroids.
 */
class IDistanceCalculator {
public:
    virtual ~IDistanceCalculator() = default;

    /**
     * @brief Computes distances between all data points and centroids.
     * @param d_data Pointer to data in device memory
     * @param d_centroids Pointer to centroids in device memory
     * @param d_outDistances Output pointer for distances (device memory)
     * @param numPoints Number of data points
     * @param numClusters Number of clusters (centroids)
     * @param dim Dimensionality of each data point/centroid
     */
    virtual void computeDistances(float* d_data,
        float* d_centroids,
        float* d_outDistances,
        int numPoints,
        int numClusters,
        int dim) = 0;
};
