#pragma once

#include "IDistanceCalculator.h"

/**
 * @class EuclideanDistanceCalculator
 * @brief Implements Euclidean distance calculation on the GPU.
 *
 * Open/Closed Principle:
 * - Extends the IDistanceCalculator interface for Euclidean distance.
 */
class EuclideanDistanceCalculator : public IDistanceCalculator {
public:
    void computeDistances(float* d_data,
        float* d_centroids,
        float* d_outDistances,
        int numPoints,
        int numClusters,
        int dim) override;
};
