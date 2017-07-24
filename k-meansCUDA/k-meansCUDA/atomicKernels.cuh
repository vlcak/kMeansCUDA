#ifndef ATOMICKERNELS_CU
#define ATOMICKERNELS_CU

#include "baseKernel.cuh"

// Thread per point - Each thread iterates through all means and then atomically add point coordinates to the future kernel sum
// dataOffset is for BIG data purposes
__global__ void findNearestClusterAtomicKernel(const my_size_t meansSize, const value_t *means, value_t *measnSums, const my_size_t dataSize, const value_t* data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);

__global__ void findNearestClusterAtomicSharedMemoryKernel(const my_size_t meansSize, const value_t* __restrict__ means, value_t *measnSums, const my_size_t dataSize, const value_t*  __restrict__ data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);

// Thread per point (Transposed data) - Each thread iterates through all means and then atomically add point coordinates to the future kernel sum
__global__ void findNearestClusterAtomicKernelTransposed(const my_size_t meansSize, const value_t *means, value_t *measnSums, const my_size_t dataSize, const value_t* data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize);

// Thread per mean (one block can count more means based on dimension) - counts new means as an avarege from the sum of assigned clusters and number of assigned points
__global__ void countDivMeansKernel(const uint32_t* counts, value_t* means, const value_t* meansSums, const my_size_t dimension, const uint32_t meansPerBlock);

#endif //ATOMICKERNELS_CU