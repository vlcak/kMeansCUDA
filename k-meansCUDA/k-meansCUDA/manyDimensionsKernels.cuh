#ifndef MANYDIMENSIONKERNELS_CU
#define MANYDIMENSIONKERNELS_CU

#include "baseKernel.cuh"

// Warp per point - threads in warp iterates through all dimensions and then total distance to mean is computed from partial distances (each thread stores partial distance in shared memory).
// Reduction can be done in loop, in unrolled loop or by shuffle instructions.
// This is done for all means and the nearest is chosen.

// Reduction is done in a loop
__global__ void findNearestClusterManyDimKernel(const my_size_t meansSize, const value_t *means, value_t *meansSums, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const my_size_t dimension);

// Reduction is done ununrolled lopp
__global__ void findNearestClusterManyDimUnrolledKernel(const my_size_t meansSize, const value_t *means, value_t *meansSums, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const my_size_t dimension);

// Reduction is done by shuffle instructions - does not need shared memory
__global__ void findNearestClusterManyDimShuffleKernel(const my_size_t meansSize, const value_t *means, value_t *meansSums, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const my_size_t dimension);

#endif //MANYDIMENSIONKERNELS_CU