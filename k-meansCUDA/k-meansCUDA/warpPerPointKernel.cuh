#ifndef WARPPERPOINTKERNELS_CU
#define WARPPERPOINTKERNELS_CU

#include "baseKernel.cuh"

__global__ void findNearestWarpPerPointKernel(const my_size_t meansSize, const value_t *means, value_t *measnSums, const value_t* data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);
__global__ void findNearestWarpPerPointSMKernel(const my_size_t meansSize, const value_t *means, value_t *measnSums, const value_t* data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);
__global__ void findNearestWarpPerPointKernelShuffle(const my_size_t meansSize, const value_t *means, value_t *measnSums, const value_t* data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);

#endif //WARPPERPOINTKERNELS_CU