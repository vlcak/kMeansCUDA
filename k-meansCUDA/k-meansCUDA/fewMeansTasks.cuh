#ifndef FEWMEANSTASKS_CUH
#define FEWMEANSTASKS_CUH

#include "baseKernel.cuh"
#include <stdint.h>

cudaError_t countKMeansFewMeans(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansFewMeansV2(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansFewMeansV3(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);

#endif //FEWMEANSTASKS_CUH