#ifndef WARPPERMEANTASKS_CUH
#define WARPPERMEANTASKS_CUH

#include "baseKernel.cuh"
#include <stdint.h>

cudaError_t countKMeansWarpPerMean(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);

#endif //WARPPERMEANTASKS_CUH