#ifndef BASEKERNEL_CUH
#define BASEKERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>

typedef float value_t;
typedef unsigned char cluster_t;

const int BLOCK_SIZE = 64;
const int WARP_SIZE = 32;
#ifdef __CUDACC__
#pragma message "using nvcc"
#ifdef __CUDA_ARCH__
#pragma message "device code trajectory"
#if __CUDA_ARCH__ < 300
#pragma message "compiling for Fermi and older"
#elif __CUDA_ARCH__ < 500
#pragma message "compiling for Kepler"
#else
#pragma message "compiling for Maxwell"
#endif
#endif
#else
#pragma message "non - nvcc code trajectory"
#endif

#endif //BASEKERNEL_H