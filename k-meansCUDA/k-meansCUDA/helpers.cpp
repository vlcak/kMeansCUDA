#include "helpers.h"

#include <iostream>

std::unique_ptr<value_t[]> loadData(const std::string& file_name, uint64_t& dataSize, uint64_t& dimension)
{
    //FILE* f = fopen(file_name.c_str(), "rb");
    std::unique_ptr<std::FILE, decltype(&std::fclose)> f(std::fopen(file_name.c_str(), "rb"), &std::fclose);
    if (!f) throw std::runtime_error("cannot open file for reading");
    //if (fseek(f, 0, SEEK_END)) throw std::runtime_error("seeking failed");
    if (!fread(&dataSize, sizeof(uint64_t), 1, f.get()))  throw std::runtime_error("size cannot be read");
    if (!fread(&dimension, sizeof(uint64_t), 1, f.get()))  throw std::runtime_error("dimension cannot be read");
    std::unique_ptr<value_t[]> data = std::make_unique<value_t[]>(dataSize * dimension); // = (value_t*)std::calloc(dataSize * dimension, sizeof(value_t));
    if (!fread(data.get(), sizeof(value_t), dataSize * dimension, f.get()))  throw std::runtime_error("value cannot be read");
    return data;
}

void saveResults(const std::string& means_file_name, const std::string& clusters_file_name, const uint32_t meansSize, std::unique_ptr<value_t[]> const &means, const uint32_t dataSize, std::unique_ptr<value_t[]> const &data, std::unique_ptr<uint32_t[]> const &assignedClusters, uint64_t& dimension)
{
    //FILE* f = fopen(means_file_name.c_str(), "wb");
    std::unique_ptr<std::FILE, decltype(&std::fclose)> f_means(std::fopen(means_file_name.c_str(), "wb"), &std::fclose);
    if (!f_means) throw std::runtime_error("cannot open file for writing");
    if (!fwrite(&dimension, sizeof(uint64_t), 1, f_means.get())) throw std::runtime_error("dimension cannot be written");
    for (size_t i = 0; i < meansSize; i++)
    {
        if (!fwrite(&means[i*dimension], sizeof(value_t), dimension, f_means.get())) throw std::runtime_error("value cannot be written");
        if (!fwrite(&i, sizeof(unsigned char), 1, f_means.get())) throw std::runtime_error("value cannot be written");
    }

    //f = fopen(clusters_file_name.c_str(), "wb");
    std::unique_ptr<std::FILE, decltype(&std::fclose)> f_clusters(std::fopen(clusters_file_name.c_str(), "wb"), &std::fclose);
    if (!f_clusters) throw std::runtime_error("cannot open file for writing");
    if (!fwrite(&dimension, sizeof(uint64_t), 1, f_clusters.get())) throw std::runtime_error("dimension cannot be written");
    for (size_t i = 0; i < dataSize; i++)
    {
        if (!fwrite(&data[i*dimension], sizeof(value_t), dimension, f_clusters.get())) throw std::runtime_error("value cannot be written");
        if (!fwrite(&assignedClusters[i], sizeof(unsigned char), 1, f_clusters.get())) throw std::runtime_error("value cannot be written");
    }
}

void verifyDeviceID(const int deviceID)
{
    int cudaDeviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&cudaDeviceCount);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("Can't detect CUDA devices! Error: " + error);
    }
    else
    {
        if (deviceID >= cudaDeviceCount)
        {
            throw std::runtime_error("Device ID not valid! Specified device ID: " + deviceID);
        }
    }
}

void printDeviceInfo(const int deviceID)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceID);
    std::cout << "Device Number: " << deviceID << std::endl;
    std::cout << "  Device name: " << prop.name << std::endl;
    std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
    std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
    std::cout << "  Shared memory per block / per Multiprocessor: " << prop.sharedMemPerBlock << "/"<< prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "  Multiprocessors count: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Warp size: " << prop.warpSize << std::endl;
    std::cout << "  Max block size: " << std::dec << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max grid size size: " << prop.maxGridSize[0] << ";" << prop.maxGridSize[1] << ";" << prop.maxGridSize[2] << std::endl;
    std::cout << std::endl;
}

void setDevice(const int deviceID)
{
    std::cout << "Set CUDA device: " << deviceID << std::endl;
    cudaError_t value = cudaSetDevice(deviceID);
    if (value != cudaSuccess)
    {
        throw CUDASetDeviceException();
    }
}

void allocateMemory(void** devicePointer, const size_t size)
{
    cudaError_t value = cudaMalloc(devicePointer, size);
    if (value != cudaSuccess)
    {
        throw CUDAMemoryAllocationException();
    }
}

void allocateAndSetMemory(void** devicePointer, const size_t size, const int valueToSet)
{
    cudaError_t value = cudaMalloc(devicePointer, size);
    if (value != cudaSuccess)
    {
        throw CUDAMemoryAllocationException();
    }
    else
    {
        value = cudaMemset(*devicePointer, valueToSet, size);
        if (value != cudaSuccess)
        {
            throw CUDAMemorySettingException();
        }
    }
}

void copyMemory(void *destination, const void *source, const size_t count, const enum cudaMemcpyKind kind)
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
        throw CUDASyncException(value);
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

