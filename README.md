# GPU-Accelerated KMeans Clustering
This project showcases a CUDA-accelerated KMeans clustering implementation in C++. By leveraging NVIDIA GPUs, you can efficiently process large datasets, explore cluster structures, and evaluate results using inertia and silhouette scores.

## Overview
### Large Dataset Support

Efficiently handle tens (or hundreds) of thousands of data points in multiple dimensions.
Distance Calculator

Uses a configurable distance metric interface (with a default Euclidean calculator).
Clustering Evaluation

Computes inertia (sum of squared distances to centroids).
Computes silhouette (a measure of how well-separated clusters are).
### Extendable & Modular

Add new distance calculators by implementing a simple interface.
Easily change the number of clusters or dimensions without rewriting the core.
Features
CUDA Kernels
Distance calculation (computeEuclideanDistancesKernel)
Cluster assignment (assignClustersKernel)
Centroid updates (updateCentroidsKernel)
SOLID Design
Each class focuses on a single concern (Dataset, DistanceCalculator, KMeans, etc.).
Easily extend or swap out components (e.g., add Manhattan distance or hierarchical clustering).
Configurable
Adjust the number of points, dimensions, clusters, and KMeans iterations directly in main.cpp.
Installation & Build
Prerequisites
NVIDIA CUDA Toolkit (tested with version 12.x or newer)
A C++17-capable compiler (e.g., MSVC 2019+, GCC 9+, or Clang 9+)
CMake (optional but recommended), or you can directly use nvcc in Visual Studio / command line.
Building with CMake (Example)
bash
Copy code
git clone <your_repository_url>.git
cd kmeans_cuda
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
Then run:

bash
Copy code
./kmeans_cuda
Building in Visual Studio
Open the project in Visual Studio.
Right-click the project → Build Customizations... → check CUDA X.X.
Ensure each .cu file has Item Type = CUDA C/C++.
Compile and run from within Visual Studio.
Usage
Adjust Dataset Size and Settings
In main.cpp, change numPoints, dim, and maxClusters to fit your data or compute budget.
Run the Application
Your console output will indicate the optimal number of clusters, final inertia, silhouette score, and execution time.
Analyze the Results
bestK shows the best cluster count according to silhouette.
bestInertia indicates how tight clusters are (lower is better).
bestSilhouette indicates how well-separated clusters are (closer to 1.0 is better).
Example Output
An example run (with numPoints = 100000, dim = 10, and maxClusters = 5) might print:

yaml
Copy code
Number of Points:  100000
Dimensions:        10
Best K:            3
Best Inertia:      5.34761e+07
Best Silhouette:   0.52
KMeans took:       8.32 seconds.
(Values vary based on random initialization and hardware.)

## Performance Tips
Increase Block Size: In the CUDA kernels, adjust dim3 block(16, 16); or your 1D block size to match your GPU architecture.
Memory Coalescing: Ensure data is stored in row-major or column-major format consistently to improve memory throughput.
k-means++ Initialization: For better cluster quality, consider implementing k-means++ rather than naive random centroids.
Profiling: Use NVIDIA Nsight or nvprof to pinpoint bottlenecks.
Contributing
Contributions are welcome! Feel free to open issues, create pull requests, or suggest new features:

New distance metrics (Manhattan, Minkowski, Cosine, etc.)
Additional clustering evaluators (Calinski-Harabasz, Davies-Bouldin)
Enhanced initialization strategies (k-means++, random partition, etc.)

## License
This project is licensed under the MIT License. You’re free to modify and redistribute it under the terms of the MIT license.

## Acknowledgments
NVIDIA for the CUDA Toolkit
Mersenne Twister for high-quality random generation
The open-source community for continuing to improve GPGPU computing tools!