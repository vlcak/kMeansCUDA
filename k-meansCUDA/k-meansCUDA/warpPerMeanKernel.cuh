#ifndef FEWMEANSKERNELS_CUH
#define FEWMEANSKERNELS_CUH

#include "baseKernel.cuh"

__global__ void findNearestClusterWarpPerMeanKernel(const value_t *means, const uint32_t dataSize, const value_t* data, uint32_t* locks, value_t* distances, uint32_t* assignedClusters, const uint32_t dimension);

__global__ void countNewMeansWarpPerMeansKernel(value_t* newMeans, const uint32_t dataSize, const value_t* data, const uint32_t* assignedClusters, const uint32_t dimension);

#endif //FEWMEANSKERNELS_CUH