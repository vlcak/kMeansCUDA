#ifndef BASEKERNEL_CUH
#define BASEKERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>

typedef float value_t;
typedef int32_t my_size_t;
typedef unsigned char cluster_t;

const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;
#ifdef __CUDACC__
#pragma message "CUDA code"
#else
#pragma message "non-CUDA code"
#endif

#ifdef __CUDA_ARCH__
#pragma message "device code trajectory"
#if __CUDA_ARCH__ < 300
#pragma message "compiling for Fermi and older"
#elif __CUDA_ARCH__ < 500
#pragma message "compiling for Kepler"
#else
#pragma message "compiling for Maxwell"
#endif
#else
#pragma message "__CUDA_ARCH__ not defined"
#endif

#endif //BASEKERNEL_H