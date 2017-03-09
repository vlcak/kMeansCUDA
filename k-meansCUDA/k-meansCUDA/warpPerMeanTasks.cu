#include "warpPerMeanTasks.cuh"
#include "warpPerMeanKernel.cuh"
#include "atomicKernels.cuh"
#include "helpers.h"

#include <time.h>
#include <stdio.h>
#include <iostream>


cudaError_t countKMeansWarpPerMean(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
	value_t *dev_means = 0, *dev_data = 0, *dev_distances = 0, *dev_temp = 0;
	uint32_t *dev_assignedClusters = 0, *dev_counts = 0, *dev_locks = 0;

	// Launch a kernel on the GPU with one thread for each element.
	dim3 blockSizeN = BLOCK_SIZE;
	int nBlocksN = meansSize;

	// for DivMeansKernel
	dim3 blockSizeMeans;
	blockSizeMeans.x = dimension;
	blockSizeMeans.y = BLOCK_SIZE / dimension;

	int gridSizeMeans = (meansSize - 1) / blockSizeMeans.y + 1;

	clock_t start, end;
	start = clock();

	cudaError_t cudaStatus;

	//std::vector<uint32_t> testVector(meansSize);

	try
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		setDevice(DEVICE_ID);

		// Allocate GPU buffers for three vectors (two input, one output)    .
		allocateMemory((void**)&dev_means, meansSize * dimension * sizeof(value_t));

		allocateAndSetMemory((void**)&dev_distances, dataSize * sizeof(value_t), UINT32_MAX);

		allocateAndSetMemory((void**)&dev_locks, dataSize * sizeof(uint32_t), UINT_MAX);

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
		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterWarpPerMeanKernel << <nBlocksN, blockSizeN >> >(dev_means, dataSize, dev_data, dev_locks, dev_distances, dev_assignedClusters, dimension);
			synchronizeDevice();
			countNewMeansWarpPerMeansKernel << <gridSizeMeans, blockSizeMeans >> >(dev_means, dataSize, dev_data, dev_assignedClusters, dimension);
			synchronizeDevice();

			cudaMemset(dev_distances, UINT32_MAX, dataSize * sizeof(value_t));
		}

		// Check for any errors launching the kernel
		checkErrors();

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		//if (cudaDeviceSynchronize() != cudaSuccess) {
		//	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		//	throw CUDASyncException(cudaStatus);
		//}

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

	// free memory
	cudaFree(dev_means);
	cudaFree(dev_distances);
	cudaFree(dev_locks);
	cudaFree(dev_data);
	cudaFree(dev_assignedClusters);
	cudaFree(dev_counts);

	end = clock();
	std::cout << "Time required for execution: "
		<< (double)(end - start) / CLOCKS_PER_SEC
		<< " seconds." << "\n\n";

	return cudaStatus;
}