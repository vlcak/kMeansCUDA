#include "cudaException.h"


const char* CUDASetDeviceException::what() const throw()
{
    return "CUDA set device failed";
}

const char* CUDAMemoryAllocationException::what() const throw()
{
    return "CUDA memory allocation failed";
}

const char* CUDAMemorySettingException::what() const throw()
{
    return "CUDA memory setting failed";
}

const char* CUDAMemoryCopyException::what() const throw()
{
    return "CUDA memory copy failed";
}

CUDASyncException::CUDASyncException(cudaError_t p_cudaError)
{
    cudaError = p_cudaError;
}

const char* CUDASyncException::what() const throw()
{
    return "CUDA device synchronization failed: " + cudaError;
}

cudaError_t CUDASyncException::getError()
{
    return cudaError;
}

const char* CUDAKernelException::what() const throw()
{
    return "CUDA kernel failed";
}

CUDAGeneralException::CUDAGeneralException(cudaError_t p_cudaError)
{
    cudaError = p_cudaError;
}
const char* CUDAGeneralException::what() const throw()
{
    return cudaGetErrorString(cudaError);
}

cudaError_t CUDAGeneralException::getError()
{
    return cudaError;
}