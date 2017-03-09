#include "cudaException.h"


const char* CUDASetDeviceException::what() const throw()
{
	return "CUDA set device failed";
}

const char* CUDAMemoryAllocationException::what() const throw()
{
	return "CUDA memory allocation failed";
}

const char*CUDAMemoryCopyException::what() const throw()
{
	return "CUDA memory copy failed";
}


const char* CUDASyncException::what() const throw()
{
	return "CUDA device synchronization failed";
}

const char* CUDAKernelException::what() const throw()
{
	return "CUDA kernel failed";
}

CUDAGeneralException::CUDAGeneralException(cudaError_t cudaErrorP)
{
	cudaError = cudaErrorP;
}
const char* CUDAGeneralException::what() const throw()
{
	return cudaGetErrorString(cudaError);
}

cudaError_t CUDAGeneralException::getError()
{
	return cudaError;
}