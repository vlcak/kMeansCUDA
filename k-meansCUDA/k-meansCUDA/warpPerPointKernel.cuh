#ifndef WARPPERPOINTKERNELS_CU
#define WARPPERPOINTKERNELS_CU

#include "baseKernel.cuh"

__global__ void findNearestWarpPerPointKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, const uint32_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);
__global__ void findNearestWarpPerPointSMKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, const uint32_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);
__global__ void findNearestWarpPerPointKernelShuffle(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, const uint32_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);

#endif //WARPPERPOINTKERNELS_CU