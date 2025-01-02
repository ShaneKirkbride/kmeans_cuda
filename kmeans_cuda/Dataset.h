#pragma once
#include <vector>

/**
 * @class Dataset
 * @brief Manages host and device memory for dataset points.
 *
 * Single Responsibility:
 * - Loads/owns the data in host memory.
 * - Allocates/frees GPU memory.
 * - Provides device pointers for other components.
 */
class Dataset {
public:
    /**
     * @brief Constructs a Dataset from a given host vector of floats.
     * @param inputData Flattened array of data points: size = numPoints * dim
     * @param numPoints Number of data points
     * @param dimension Dimensionality of each point
     */
    Dataset(const std::vector<float>& inputData, int numPoints, int dimension);
    ~Dataset();

    int getNumPoints() const;
    int getDim() const;

    /**
     * @return Pointer to data on the device
     */
    float* getDeviceData() const;

private:
    // Host data: flattened (x1, x2, ..., xD, x1, x2,...)
    std::vector<float> h_data_;

    // Device data pointer
    float* d_data_;

    int numPoints_;
    int dim_;

    void allocateDeviceMemory();
    void freeDeviceMemory();
};
