#ifndef MANYDIMENSIONKERNELS_CU
#define MANYDIMENSIONKERNELS_CU

#include "baseKernel.h"

__global__ void findNearestClusterManyDimKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension);

__global__ void findNearestClusterManyDimUnrolledKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension, const uint16_t blockSize);

__global__ void findNearestClusterManyDimShuffleKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension);

#endif //MANYDIMENSIONKERNELS_CU