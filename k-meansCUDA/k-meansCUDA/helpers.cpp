#include "helpers.h"

void setDevice(int deviceID)
{
	cudaError_t value = cudaSetDevice(deviceID);
	if (value != cudaSuccess)
	{
		throw CUDASetDeviceException();
	}
}

void allocateMemory(void** devicePointer, size_t size)
{
	cudaError_t value = cudaMalloc(devicePointer, size);
	if (value != cudaSuccess)
	{
		throw CUDAMemoryAllocationException();
	}
}

void allocateAndSetMemory(void** devicePointer, size_t size, int valueToSet)
{
	cudaError_t value = cudaMalloc(devicePointer, size);
	if (value != cudaSuccess)
	{
		throw CUDAMemoryAllocationException();
	}
	else
	{
		cudaMemset(devicePointer, valueToSet, size);
	}
}

void copyMemory(void *destination, const void *source, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t value = cudaMemcpy(destination, source, count, kind);
	if (value != cudaSuccess)
	{
		throw CUDAMemoryCopyException();
	}
}

void synchronizeDevice()
{
	cudaError_t value = cudaDeviceSynchronize();
	if (value != cudaSuccess)
	{
		throw CUDASyncException();
	}
}

cudaError_t checkErrors()
{
	cudaError_t value = cudaGetLastError();
	if (value != cudaSuccess)
	{
		throw CUDAGeneralException(value);
	}
	return value;
}

