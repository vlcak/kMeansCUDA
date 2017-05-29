#ifndef SIMPLEKERNELS_CU
#define SIMPLEKERNELS_CU

#include "baseKernel.cuh"

__global__ void findNearestClusterKernel(const my_size_t meansSize, const value_t *means, const value_t* data, uint32_t* assignedClusters, const my_size_t dimension);

__global__ void countNewMeansKernel(uint32_t* assignedClusters, const my_size_t dataSize, const value_t* data, value_t* means, const my_size_t dimension, uint32_t* test);

#endif //SIMPLEKERNELS_CU