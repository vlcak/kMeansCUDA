#include "warpPerPointTasks.cuh"
#include "warpPerPointKernel.cuh"
#include "atomicKernels.cuh"
#include "helpers.h"

#include <time.h>
#include <stdio.h>
#include <iostream>


cudaError_t countKMeansWarpPerPoint(const uint32_t iterations, const uint32_t dataSize_u32, const value_t* data, const uint32_t meansSize_u32, value_t* means, uint32_t* assignedClusters, uint64_t dimension_u64, std::string version)
{
    value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0;//, *dev_temp = 0;
    uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
    const my_size_t dataSize = static_cast<my_size_t>(dataSize_u32);
    const my_size_t meansSize = static_cast<my_size_t>(meansSize_u32);
    const my_size_t dimension = static_cast<my_size_t>(dimension_u64);
    cudaError_t cudaStatus = cudaSuccess;

    // Launch a kernel on the GPU with one thread for each element.
    int pointsPerWarp = BLOCK_SIZE / meansSize;
    dim3 blockSizeN(meansSize, pointsPerWarp);
    int nBlocksN = (dataSize - 1) / pointsPerWarp + 1;
    auto findNearestClusterKernel = &findNearestWarpPerPointKernel;
    int sharedMemomrySize = sizeof(value_t)* (/*dimension * pointsPerWarp + */blockSizeN.x * blockSizeN.y);
    if (version == "--sharedMemory")
    {
        findNearestClusterKernel = &findNearestWarpPerPointSMKernel;
        sharedMemomrySize = sizeof(value_t)* (dimension * pointsPerWarp + blockSizeN.x * blockSizeN.y);
        std::cout << "Shared memory" << std::endl;
    }
#if __CUDA_ARCH__ >= 300
    if (version == "--shuffle")
    {
        findNearestClusterKernel = &findNearestWarpPerPointShuffleKernel;
        sharedMemomrySize = 0;
        std::cout << "Shuffle" << std::endl;
    }
#endif

    // for DivMeansKernel
    int meansPerBlock = BLOCK_SIZE / dimension;
    int meansBlocks = (meansSize - 1) / meansPerBlock + 1;


    clock_t start, end;
    start = clock();

    //std::vector<uint32_t> testVector(meansSize);

    try
    {
        // Choose which GPU to run on, change this on a multi-GPU system.
        setDevice(DEVICE_ID);

        // Allocate GPU buffers for three vectors (two input, one output)    .
        allocateMemory((void**)&dev_means, meansSize * dimension * sizeof(value_t));

        allocateAndSetMemory((void**)&dev_meansSums, meansSize * dimension * sizeof(value_t), 0);

        allocateMemory((void**)&dev_data, dataSize * dimension * sizeof(value_t));

        allocateMemory((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));

        allocateAndSetMemory((void**)&dev_counts, meansSize * sizeof(uint32_t), 0);

        // Copy input vectors from host memory to GPU buffers.
        copyMemory(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
        copyMemory(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);

        //uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
        //value_t* testMeans = (value_t*)calloc(meansSize * dimension , sizeof(value_t));

        //int blockSizeM = 16;
        //int nBlocksM = (meansSize - 1) / blockSizeM + 1;

        std::cout << "Starting execution" << std::endl;
        for (int32_t i = 0; i < iterations; ++i)
        {
            findNearestClusterKernel << <nBlocksN, blockSizeN, sharedMemomrySize >> >(dev_means, dev_meansSums, dev_data, dev_counts, dimension);
            synchronizeDevice();
            countDivMeansKernel << <meansBlocks, meansPerBlock * dimension >> >(dev_counts, dev_means, dev_meansSums, dimension, meansPerBlock);
            synchronizeDevice();

            cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
            cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
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
    cudaFree(dev_meansSums);
    cudaFree(dev_assignedClusters);
    cudaFree(dev_counts);

    end = clock();
    std::cout << "Time required for execution: "
        << (double)(end - start) / CLOCKS_PER_SEC
        << " seconds." << "\n\n";

    return cudaStatus;
}