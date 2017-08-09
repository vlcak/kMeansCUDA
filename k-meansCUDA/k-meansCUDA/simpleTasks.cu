#include "simpleTasks.cuh"
#include "simpleKernels.cuh"
#include "helpers.h"

#include <time.h>
#include <stdio.h>
#include <iostream>

cudaError_t countKMeansSimple(const uint32_t iterations, const uint32_t dataSize_u32, const value_t* data, const uint32_t meansSize_u32, value_t* means, uint32_t* assignedClusters, uint64_t dimension_u64)
{
    value_t* dev_means = 0;
    value_t* dev_data = 0;
    uint32_t* dev_assignedClusters = 0;
    const my_size_t dataSize = static_cast<my_size_t>(dataSize_u32);
    const my_size_t meansSize = static_cast<my_size_t>(meansSize_u32);
    const my_size_t dimension = static_cast<my_size_t>(dimension_u64);
    cudaError_t cudaStatus = cudaSuccess;

    // Launch a kernel on the GPU with one thread for each element.
    int blockSizeN = BLOCK_SIZE;
    int nBlocksN = (dataSize - 1) / blockSizeN + 1;
	int meansPerBlock = BLOCK_SIZE > dimension ? BLOCK_SIZE / dimension : 1;
    dim3 blockSizeM(dimension, meansPerBlock);
    int nBlocksM = (meansSize - 1) / blockSizeM.y + 1;

    clock_t start, end;
    start = clock();

    try
    {
        // Choose which GPU to run on, change this on a multi-GPU system.
        setDevice(DEVICE_ID);

        // Allocate GPU buffers for three vectors (two input, one output)    .
        allocateMemory((void**)&dev_means, meansSize * dimension * sizeof(value_t));

        allocateMemory((void**)&dev_data, dataSize * dimension * sizeof(value_t));

        allocateMemory((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));

        // Copy input vectors from host memory to GPU buffers.
        copyMemory(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
        copyMemory(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);

        std::cout << "Starting execution" << std::endl;
        for (uint32_t i = 0; i < iterations; ++i)
        {
            findNearestClusterKernel << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dev_data, dev_assignedClusters, dimension);
            synchronizeDevice();
            countNewMeansKernel << <nBlocksM, blockSizeM >> >(dev_assignedClusters, dataSize, dev_data, dev_means, dimension);
            synchronizeDevice();
            //std::vector<uint32_t> t(test, test + meansSize);
        }

        // Check for any errors launching the kernel
        checkErrors();

        copyMemory(means, dev_means, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
        copyMemory(assignedClusters, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    }
    catch (ICUDAException &e)
    {
        fprintf(stderr, "CUDA exception: %s\n", e.what());
        cudaStatus = e.getError();
    }
    catch (std::exception &e)
    {
        fprintf(stderr, "STD exception: %s\n", e.what());
        cudaStatus = cudaGetLastError();
    }

    cudaFree(dev_data);
    cudaFree(dev_means);
    cudaFree(dev_assignedClusters);

    end = clock();
    std::cout << "Time required for execution: "
        << (double)(end - start) / CLOCKS_PER_SEC
        << " seconds." << "\n\n";

    return cudaStatus;
}