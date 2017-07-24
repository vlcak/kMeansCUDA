#ifndef FEWMEANSKERNELS_CUH
#define FEWMEANSKERNELS_CUH

#include "baseKernel.cuh"

// Warp per mean - each mean is copied to local memory and then each thread computes distance to one point.
// When distance to point is less than the shortest know distance, mean id is stored to assignedClusters and distance to distances - access synchronization is done through locks array (default zero, when lock meanID)
__global__ void findNearestClusterWarpPerMeanThreadPerPointKernel(const value_t *means, const my_size_t dataSize, const value_t* data, uint32_t* locks, value_t* distances, uint32_t* assignedClusters, const my_size_t dimension);

// Warp per mean - each mean is copied to local memory and then each thread computes distance in single dimension.
// When distance to point is less than the shortest know distance, mean id is stored to assignedClusters and distance to distances - access synchronization is done through locks array (default zero, when lock meanID)
__global__ void findNearestClusterWarpPerMeanThreadPerDimensionKernel(const value_t *means, const my_size_t dataSize, const value_t* data, uint32_t* locks, value_t* distances, uint32_t* assignedClusters, const my_size_t dimension);

// Warp per mean - new means are computed by going thourgh all points are if assigned mean is mean computed by current warp, it is added to coordinates sum
// Thread per dimension, block can compute multiple means (based on dimension)
__global__ void countNewMeansWarpPerMeansKernel(value_t* newMeans, const my_size_t dataSize, const value_t* data, const uint32_t* assignedClusters, const my_size_t dimension);

#endif //FEWMEANSKERNELS_CUH