#ifndef WARPPERPOINTKERNELS_CU
#define WARPPERPOINTKERNELS_CU

#include "baseKernel.cuh"

// Block per point kernel - wach thread computes distance to one mean and then the nearest mean is found. Distances to each mean are stored in shared memory
// At the end, mean sums are updated in global memory
__global__ void findNearestWarpPerPointKernel(const value_t *means, value_t *measnSums, const value_t* data, uint32_t* counts, const my_size_t dimension);

// Block per point kernel - wach thread computes distance to one mean and then the nearest mean is found.
// Point is stored in shared memory. Distances to each mean are stored in shared memory
// At the end, mean sums are updated in global memory
__global__ void findNearestWarpPerPointSMKernel(const value_t *means, value_t *measnSums, const value_t* data, uint32_t* counts, const my_size_t dimension);

// Block per point kernel - wach thread computes distance to one mean and then the nearest mean is found. Nearest mean is found using shuffle functions
// At the end, mean sums are updated in global memory
__global__ void findNearestWarpPerPointShuffleKernel(const value_t *means, value_t *measnSums, const value_t* data, uint32_t* counts, const my_size_t dimension);

#endif //WARPPERPOINTKERNELS_CU