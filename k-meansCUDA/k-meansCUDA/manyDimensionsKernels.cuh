#ifndef MANYDIMENSIONKERNELS_CU
#define MANYDIMENSIONKERNELS_CU

#include "baseKernel.cuh"

__global__ void findNearestClusterManyDimKernel(const my_size_t meansSize, const value_t *means, value_t *meansSums, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const my_size_t dimension);

__global__ void findNearestClusterManyDimUnrolledKernel(const my_size_t meansSize, const value_t *means, value_t *meansSums, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const my_size_t dimension);

// does not need shared memory
__global__ void findNearestClusterManyDimShuffleKernel(const my_size_t meansSize, const value_t *means, value_t *meansSums, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const my_size_t dimension);

#endif //MANYDIMENSIONKERNELS_CU