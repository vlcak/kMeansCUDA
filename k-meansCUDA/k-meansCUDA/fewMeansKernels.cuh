#ifndef FEWMEANSKERNELS_CUH
#define FEWMEANSKERNELS_CUH

#include "baseKernel.cuh"

// n copies of means, copy is determined by threadId.y
__global__ void findNearestClusterFewMeansKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension);

__global__ void countDivFewMeansKernel(const uint32_t meansSize, uint32_t* counts, value_t* means, const value_t* meansSums, const uint32_t dimension, const uint32_t cellsCount);

// each thread has own copy...
__global__ void findNearestClusterFewMeansKernelV2(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension);

__global__ void countDivFewMeansKernelV2(const uint32_t dataSize, const uint32_t meansSize, uint32_t* counts, value_t* means, value_t* meansSums, const uint32_t dimension);

__global__ void findNearestClusterFewMeansKernelV3(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension);

__global__ void countDivFewMeansKernelV3(const uint32_t dataSize, const uint32_t meansSize, uint32_t* counts, value_t* means, value_t* meansSums, const uint32_t dimension);

#endif //FEWMEANSKERNELS_CUH