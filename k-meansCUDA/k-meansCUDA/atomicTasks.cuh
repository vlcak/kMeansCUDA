#ifndef ATOMICTASKS_CUH
#define ATOMICTASKS_CUH

#include "baseKernel.cuh"
#include <stdint.h>

cudaError_t countKMeansAtomic(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansAtomicTransposed(const uint32_t iterations, const uint32_t dataSize, value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansBIGDataAtomic(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);

#endif //ATOMICTASKS_CUH