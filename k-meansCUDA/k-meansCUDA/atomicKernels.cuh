#ifndef ATOMICKERNELS_CU
#define ATOMICKERNELS_CU

#include "baseKernel.cuh"

__global__ void findNearestClusterAtomicKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, const uint32_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);

__global__ void findNearestClusterAtomicKernelTransposed(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, const uint32_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);

__global__ void countDivMeansKernel(const uint32_t meansSize, const uint32_t* counts, value_t* means, const value_t* meansSums, const uint32_t dimension, const uint32_t meansPerBlock);

#endif //ATOMICKERNELS_CU