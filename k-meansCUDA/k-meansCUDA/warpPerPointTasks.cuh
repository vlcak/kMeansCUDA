#ifndef WARPPERPOINTTASKS_CUH
#define WARPPERPOINTTASKS_CUH

#include "baseKernel.cuh"
#include <stdint.h>

cudaError_t countKMeansWarpPerPoint(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);

#endif //WARPPERPOINTTASKS_CUH