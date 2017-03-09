#include "simpleTasks.cuh"
#include "simpleKernels.cuh"
#include "helpers.h"

#include <time.h>
#include <stdio.h>
#include <iostream>

cudaError_t countKMeansSimple(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
	value_t* dev_means = 0;
	value_t* dev_data = 0;
	uint32_t* dev_assignedClusters = 0, *dev_test = 0;
	cudaError_t cudaStatus;

	// Launch a kernel on the GPU with one thread for each element.
	int blockSizeN = BLOCK_SIZE;
	int nBlocksN = (dataSize - 1) / blockSizeN + 1;
	int blockSizeM = 16;
	int nBlocksM = (meansSize - 1) / blockSizeM + 1;

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

		//uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));

		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterKernel << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dataSize, dev_data, dev_assignedClusters, dimension);
			synchronizeDevice();
			countNewMeansKernel << <nBlocksM, blockSizeM >> >(dev_assignedClusters, dataSize, dev_data, dev_means, dimension, dev_test);
			synchronizeDevice();
			//cudaStatus = cudaMemcpy(test, dev_test, meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(test, test + meansSize);
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
	cudaFree(dev_assignedClusters);

	end = clock();
	std::cout << "Time required for execution: "
		<< (double)(end - start) / CLOCKS_PER_SEC
		<< " seconds." << "\n\n";

	return cudaStatus;
}