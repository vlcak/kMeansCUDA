#ifndef CUDA_EXCEPTION_H
#define CUDA_EXCEPTION_H

#include "cuda_runtime.h"
#include <exception>

class ICUDAException : public std::exception
{
public:
    virtual const char* what() const throw() = 0;
    virtual cudaError_t getError() { return cudaSuccess; }
};

class CUDASetDeviceException : public ICUDAException
{
public:
    virtual const char* what() const throw();
};

class CUDAMemoryAllocationException : public ICUDAException
{
public:
    virtual const char* what() const throw();
};

class CUDAMemorySettingException : public ICUDAException
{
public:
    virtual const char* what() const throw();
};

class CUDAMemoryCopyException : public ICUDAException
{
public:
    virtual const char* what() const throw();
};

class CUDASyncException : public ICUDAException
{
public:
    CUDASyncException(cudaError_t cudaError);
    virtual const char* what() const throw();
    cudaError_t getError();
private:
    cudaError_t cudaError;
};

class CUDAKernelException : public ICUDAException
{
public:
    virtual const char* what() const throw();
};

class CUDAGeneralException : public ICUDAException
{
public:
    CUDAGeneralException(cudaError_t cudaError);
    virtual const char* what() const throw() override;
    cudaError_t getError();
private:
    cudaError_t cudaError;
};

#endif //CUDA_EXCEPTION_H