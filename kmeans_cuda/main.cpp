#include <iostream>
#include <memory>
#include <vector>
#include <random>       // For std::mt19937 and std::uniform_real_distribution
#include <chrono>       // For timing (optional)

#include "Dataset.h"
#include "EuclideanDistanceCalculator.h"
#include "KMeans.h"

int main()
{
    // ---------------------------
    // 1. Dataset Configuration
    // ---------------------------
    // Increase these values to make the dataset larger and more GPU-intensive.
    const int numPoints = 100000;  // Number of data points
    const int dim = 10;      // Dimensions per point
    const int maxClusters = 10;       // We'll try k in [2..5]
    const int maxIters = 10;      // KMeans iterations

    // ---------------------------
    // 2. Random Data Generation
    // ---------------------------
    // We'll create a random dataset using the Mersenne Twister engine.
    std::mt19937 gen(2501);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    std::vector<float> sampleData(numPoints * dim);
    for (int i = 0; i < numPoints; ++i) {
        for (int d = 0; d < dim; d++) {
            sampleData[i * dim + d] = dist(gen);
        }
    }

    // ---------------------------
    // 3. Create Dataset
    // ---------------------------
    Dataset dataset(sampleData, numPoints, dim);

    // ---------------------------
    // 4. Prepare KMeans
    // ---------------------------
    // Inject EuclideanDistanceCalculator
    auto distanceCalc = std::make_unique<EuclideanDistanceCalculator>();
    KMeans kmeans(std::move(distanceCalc));

    // Optional: measure the time
    auto startTime = std::chrono::high_resolution_clock::now();

    // ---------------------------
    // 5. Run KMeans
    // ---------------------------
    // We'll search for the best k in [2..maxClusters], 
    // measuring inertia and silhouette each time.
    auto [bestK, bestInertia, bestSil] = kmeans.findOptimalClusters(dataset,
        maxClusters,
        maxIters);

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    // ---------------------------
    // 6. Results
    // ---------------------------
    std::cout << "Number of Points:  " << numPoints << "\n";
    std::cout << "Dimensions:        " << dim << "\n";
    std::cout << "Best K:            " << bestK << "\n";
    std::cout << "Best Inertia:      " << bestInertia << "\n";
    std::cout << "Best Silhouette:   " << bestSil << "\n";
    std::cout << "KMeans took:       " << elapsed.count() << " seconds.\n";

    return 0;
}
