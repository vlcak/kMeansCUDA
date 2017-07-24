#ifndef SIMPLEKERNELS_CU
#define SIMPLEKERNELS_CU

#include "baseKernel.cuh"

// Thread per point - find the nearest mean for each point and stores its id to assigned clusters
__global__ void findNearestClusterKernel(const my_size_t meansSize, const value_t *means, const value_t* data, uint32_t* assignedClusters, const my_size_t dimension);

// Thread per mean coordinate - each thread computes new coordinate by one mean (dimension - threadId.x, mean - threadId.y)
// One block can compute multiple means
__global__ void countNewMeansKernel(const uint32_t* assignedClusters, const my_size_t dataSize, const value_t* data, value_t* means, const my_size_t dimension);

#endif //SIMPLEKERNELS_CU