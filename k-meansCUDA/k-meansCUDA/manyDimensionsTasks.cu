#include "manyDimensionsTasks.cuh"
#include "manyDimensionsKernels.cuh"
#include "atomicKernels.cuh"
#include "helpers.h"

#include <time.h>
#include <stdio.h>
#include <iostream>

cudaError_t countKMeansManyDims(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize_u32, value_t* means, uint32_t* assignedClusters, uint64_t dimension_u64)
{
	value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
	uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
	const my_size_t dimension = static_cast<my_size_t>(dimension_u64);
	const my_size_t meansSize = static_cast<my_size_t>(meansSize_u32);
	cudaError_t cudaStatus;

	const int blockSizeN = BLOCK_SIZE;
	const int pointsPerBlock = BLOCK_SIZE / WARP_SIZE;
	const int nBlocksN = (dataSize - 1) / pointsPerBlock + 1;
	dim3 blockGrid(WARP_SIZE, pointsPerBlock);

	// for DivMeansKernel
	int meansPerBlock = BLOCK_SIZE / dimension;
	int meansBlocks = (meansSize - 1) / meansPerBlock + 1;

	//uint32_t* testAssigned,* testCounts;
	//value_t* testDistances;

	//testAssigned = (uint32_t*)malloc(dataSize * sizeof(uint32_t));
	//testCounts = (uint32_t*)malloc(meansSize * sizeof(uint32_t));
	//testDistances = (value_t*)malloc(meansSize * dimension * sizeof(value_t));

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

		// Launch a kernel on the GPU with one thread for each element.
		//int nBlocksN = (dataSize - 1) / blockSizeN + 1;
		//int blockSizeM = 16;
		//int nBlocksM = (meansSize - 1) / blockSizeM + 1;
		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterManyDimUnrolledKernel << <nBlocksN, blockGrid, sizeof(value_t)* blockSizeN >> >(meansSize, dev_means, dev_meansSums, dev_data, dev_counts, dev_assignedClusters, dimension);
			//findNearestClusterManyDimShuffleKernel << <nBlocksN, blockGrid>> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dev_assignedClusters, dimension);
			synchronizeDevice();
			//cudaMemcpy(testAssigned, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(testAssigned, testAssigned + dataSize);
			//cudaMemcpy(testDistances, dev_meansSums, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
			//std::vector<value_t> t2(testDistances, testDistances + meansSize);
			//cudaMemcpy(testCounts, dev_counts, meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t3(testCounts, testCounts+ meansSize);

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
	catch (CUDAGeneralException &e)
	{
		fprintf(stderr, "CUDA exception: %s\n", e.what());
		cudaStatus = e.getError();
	}
	catch (std::exception &e)
	{
		fprintf(stderr, "CUDA exception: %s\n", e.what());
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