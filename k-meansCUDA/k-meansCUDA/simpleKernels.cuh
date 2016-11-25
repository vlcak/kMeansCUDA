#ifndef SIMPLEKERNELS_CU
#define SIMPLEKERNELS_CU

#include "baseKernel.h"

__global__ void findNearestClusterKernel(const uint32_t meansSize, const value_t *means, const uint32_t dataSize, const value_t* data, uint32_t* assignedClusters, const uint32_t dimension);

__global__ void countNewMeansKernel(uint32_t* assignedClusters, const uint32_t dataSize, const value_t* data, value_t* means, const uint32_t dimension, uint32_t* test);

#endif //SIMPLEKERNELS_CU