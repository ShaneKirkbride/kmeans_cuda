#pragma once
#include <memory>
#include <tuple>
#include "IDistanceCalculator.h"
#include "InertiaCalculator.h"
#include "SilhouetteCalculator.h"

class Dataset;

/**
 * @class KMeans
 * @brief Coordinates the entire KMeans clustering process.
 *
 * Single Responsibility:
 * - Manages KMeans iteration, updating centroids, and calling distance/inertia/silhouette.
 * Dependency Inversion:
 * - KMeans depends on the abstract IDistanceCalculator rather than a concrete implementation.
 */
class KMeans {
public:
    /**
     * @brief Constructor injecting a distance calculator (DIP).
     */
    KMeans(std::unique_ptr<IDistanceCalculator> distCalc);

    /**
     * @brief Tries different k values from [2..maxK], returns best (k, inertia, silhouette).
     * @param dataset Reference to the Dataset
     * @param maxK Max number of clusters to test
     * @param maxIterations Number of KMeans iterations per cluster setting
     */
    std::tuple<int, float, float> findOptimalClusters(Dataset& dataset, int maxK, int maxIterations = 10);

private:
    std::unique_ptr<IDistanceCalculator> distanceCalculator_;
    InertiaCalculator inertiaCalculator_;
    SilhouetteCalculator silhouetteCalculator_;

    /**
     * @brief Runs KMeans for a fixed k. Returns (inertia, silhouette).
     */
    std::pair<float, float> runKMeans(Dataset& dataset, int k, int maxIterations);

};
