#include "helpers.h"

#include <iostream>

value_t* loadData(const std::string& file_name, uint64_t& dataSize, uint64_t& dimension)
{
	FILE* f = fopen(file_name.c_str(), "rb");
	if (!f) throw std::runtime_error("cannot open file for reading");
	//if (fseek(f, 0, SEEK_END)) throw std::runtime_error("seeking failed");
	if (!fread(&dataSize, sizeof(uint64_t), 1, f))  throw std::runtime_error("size cannot be read");
	if (!fread(&dimension, sizeof(uint64_t), 1, f))  throw std::runtime_error("dimension cannot be read");
	value_t* data = (value_t*)calloc(dataSize * dimension, sizeof(value_t));
	if (!fread(data, sizeof(value_t), dataSize * dimension, f))  throw std::runtime_error("value cannot be read");
	return data;
}

void saveResults(const std::string& means_file_name, const std::string& clusters_file_name, const uint32_t meansSize, const value_t* means, const uint32_t dataSize, const value_t* data, const uint32_t* assignedClusters, uint64_t& dimension)
{
	FILE* f = fopen(means_file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	if (!fwrite(&dimension, sizeof(uint64_t), 1, f)) throw std::runtime_error("dimension cannot be written");
	//if (!fwrite(means, sizeof(value_t), dimension * meansSize, f)) throw std::runtime_error("value cannot be written");
	for (size_t i = 0; i < meansSize; i++)
	{
		if (!fwrite(&means[i*dimension], sizeof(value_t), dimension, f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&i, sizeof(unsigned char), 1, f)) throw std::runtime_error("value cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");


	f = fopen(clusters_file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	if (!fwrite(&dimension, sizeof(uint64_t), 1, f)) throw std::runtime_error("dimension cannot be written");
	for (size_t i = 0; i < dataSize; i++)
	{
		if (!fwrite(&data[i*dimension], sizeof(value_t), dimension, f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&assignedClusters[i], sizeof(unsigned char), 1, f)) throw std::runtime_error("value cannot be written");
		//if (!fwrite(&i, sizeof(value_t), 1, f)) throw std::runtime_error("distance cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");
}

bool verifyDeviceID(int deviceID)
{
	int cudaDeviceCount = 0;
	cudaError_t error = cudaGetDeviceCount(&cudaDeviceCount);
	return error != cudaSuccess;
}

void setDevice(int deviceID)
{
	std::cout << "Set CUDA device: " << deviceID << std::endl;
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

