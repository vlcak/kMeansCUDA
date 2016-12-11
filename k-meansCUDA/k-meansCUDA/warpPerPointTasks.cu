#include "warpPerPointTasks.cuh"
#include "warpPerPointKernel.cuh"
#include "atomicKernels.cuh"

#include <time.h>
#include <stdio.h>
#include <iostream>


cudaError_t countKMeansWarpPerPoint(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
	value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
	uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
	cudaError_t cudaStatus;

	// Launch a kernel on the GPU with one thread for each element.
	int pointsPerWarp = BLOCK_SIZE / meansSize;
	dim3 blockSizeN = (meansSize, pointsPerWarp);
	int nBlocksN = (dataSize - 1) / pointsPerWarp + 1;

	// for DivMeansKernel
	int meansPerBlock = BLOCK_SIZE / dimension;
	int meansBlocks = (meansSize - 1) / meansPerBlock + 1;

	int sharedMemomrySize = sizeof(value_t)* (dimension * pointsPerWarp + blockSizeN.x * blockSizeN.y);

	clock_t start, end;
	start = clock();

	//std::vector<uint32_t> testVector(meansSize);

	try
	{
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			throw 1;
		}

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)&dev_means, meansSize * dimension * sizeof(value_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}

		cudaStatus = cudaMalloc((void**)&dev_meansSums, meansSize * dimension * sizeof(value_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}
		else
		{
			cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
		}

		cudaStatus = cudaMalloc((void**)&dev_data, dataSize * dimension * sizeof(value_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}

		cudaStatus = cudaMalloc((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}

		cudaStatus = cudaMalloc((void**)&dev_counts, meansSize * sizeof(uint32_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}
		else
		{
			cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw 1;
		}

		cudaStatus = cudaMemcpy(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw 1;
		}

		//uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
		//value_t* testMeans = (value_t*)calloc(meansSize * dimension , sizeof(value_t));

		//int blockSizeM = 16;
		//int nBlocksM = (meansSize - 1) / blockSizeM + 1;
		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestWarpPerPointKernel << <nBlocksN, blockSizeN, sharedMemomrySize >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dimension, 0, dataSize);
			cudaDeviceSynchronize();
			countDivMeansKernel << <meansBlocks, meansPerBlock * dimension >> >(meansSize, dev_counts, dev_means, dev_meansSums, dimension, meansPerBlock);
			cudaDeviceSynchronize();

			cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
			cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
		}

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			throw 1;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			throw 1;
		}

		cudaStatus = cudaMemcpy(means, dev_means, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw 1;
		}

		cudaStatus = cudaMemcpy(assignedClusters, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw 1;
		}
	}
	catch (...)
	{
		cudaFree(dev_data);
		cudaFree(dev_means);
		cudaFree(dev_meansSums);
		cudaFree(dev_assignedClusters);
		cudaFree(dev_counts);
	}

	end = clock();
	std::cout << "Time required for execution: "
		<< (double)(end - start) / CLOCKS_PER_SEC
		<< " seconds." << "\n\n";

	return cudaStatus;
}