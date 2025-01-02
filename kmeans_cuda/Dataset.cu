#include "Dataset.h"
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>

// Simple GPU Error Check Macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
            << " " << file << ":" << line << std::endl;
        exit(code);
    }
}

Dataset::Dataset(const std::vector<float>& inputData, int numPoints, int dimension)
    : h_data_(inputData),
    d_data_(nullptr),
    numPoints_(numPoints),
    dim_(dimension)
{
    // Basic sanity check
    assert(static_cast<int>(h_data_.size()) == numPoints_ * dim_);
    allocateDeviceMemory();
}

Dataset::~Dataset() {
    freeDeviceMemory();
}

int Dataset::getNumPoints() const {
    return numPoints_;
}

int Dataset::getDim() const {
    return dim_;
}

float* Dataset::getDeviceData() const {
    return d_data_;
}

void Dataset::allocateDeviceMemory() {
    size_t dataSize = sizeof(float) * h_data_.size();
    CUDA_CHECK(cudaMalloc((void**)&d_data_, dataSize));
    CUDA_CHECK(cudaMemcpy(d_data_, h_data_.data(), dataSize, cudaMemcpyHostToDevice));
}

void Dataset::freeDeviceMemory() {
    if (d_data_) {
        CUDA_CHECK(cudaFree(d_data_));
        d_data_ = nullptr;
    }
}
