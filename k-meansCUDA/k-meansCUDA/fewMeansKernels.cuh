#ifndef FEWMEANSKERNELS_CUH
#define FEWMEANSKERNELS_CUH

#include "baseKernel.cuh"

// Thread per point - Each thread iterates through all means and then atomically add point coordinates the n-th copy of new means. Copy is determined by threadId.y
__global__ void findNearestClusterFewMeansKernel(const my_size_t meansSize, const value_t *means, value_t *measnSums, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const my_size_t dimension);

// Thread per mean - Each thread iterates through n copies of new means sums and then counts new mean
__global__ void countDivFewMeansKernel(const my_size_t meansSize, uint32_t* counts, value_t* means, const value_t* meansSums, const my_size_t dimension, const uint32_t cellsCount);

// Thread per point - Each thread iterates through all means and then add point coordinates to own copy of new means.
__global__ void findNearestClusterFewMeansKernelV2(const my_size_t meansSize, const value_t *means, value_t *measnSums, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const my_size_t dimension);

// Each thread iterates through all copies of new means sums and then counts new mean
__global__ void countDivFewMeansKernelV2(const my_size_t dataSize, const my_size_t meansSize, uint32_t* counts, value_t* means, value_t* meansSums, const my_size_t dimension);

// Thread per point - Each thread iterates through all means and then atomically add point coordinates the n-th copy of new means. Copy is determined by threadId.y. Copy is placed in the shared memory which is organized as AoS
// Then, copies are add together and saved to global memory. There is also redundancy and copy is added to m-th copy of new means based on blockid.x
__global__ void findNearestClusterFewMeansKernelV3(const my_size_t meansSize, const value_t *means, value_t *measnSums, const my_size_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const my_size_t dimension);

// Thread per point - Each thread iterates through all means and then atomically add point coordinates the n-th copy of new means. Copy is determined by threadId.y. Copy is placed in the shared memory which is organized as SoA
// Then, copies are add together and saved to global memory. There is also redundancy and copy is added to m-th copy of new means based on blockid.x
__global__ void findNearestClusterFewMeansSharedTransposedKernelV3(const my_size_t meansSize, const value_t *means, value_t *measnSums, const my_size_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const my_size_t dimension);

// Multiple threads per mean - thread per each dimension of each mean copy, block can be shared by multiple means (if redundancy * dimension is < BLOCK_SIZE / 2)
// Computes new means by adding all redundant copies together and than counting new mean as average
__global__ void countDivFewMeansKernelV3(const my_size_t meansSize, uint32_t* counts, value_t* means, value_t* meansSums, const my_size_t dimension);

#endif //FEWMEANSKERNELS_CUH