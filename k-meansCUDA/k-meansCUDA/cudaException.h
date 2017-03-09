#ifndef CUDA_EXCEPTION_H
#define CUDA_EXCEPTION_H

#include "cuda_runtime.h"
#include <exception>

class CUDASetDeviceException : public std::exception
{
	virtual const char* what() const throw();
};

class CUDAMemoryAllocationException : public std::exception
{
	virtual const char* what() const throw();
};

class CUDAMemoryCopyException : public std::exception
{
	virtual const char* what() const throw();
};

class CUDASyncException : public std::exception
{
	virtual const char* what() const throw();
};

class CUDAKernelException : public std::exception
{
	virtual const char* what() const throw();
};

class CUDAGeneralException : public std::exception
{
public:
	CUDAGeneralException(cudaError_t cudaErrorP);
	virtual const char* what() const throw() override;
	cudaError_t getError();
private:
	cudaError_t cudaError;
};

#endif //CUDA_EXCEPTION_H