#include "simpleTasks.cuh"
#include "simpleKernels.cuh"

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

		cudaStatus = cudaMalloc((void**)&dev_test, meansSize * sizeof(uint32_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
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

		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterKernel << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dataSize, dev_data, dev_assignedClusters, dimension);
			cudaDeviceSynchronize();
			countNewMeansKernel << <nBlocksM, blockSizeM >> >(dev_assignedClusters, dataSize, dev_data, dev_means, dimension, dev_test);
			cudaDeviceSynchronize();
			//cudaStatus = cudaMemcpy(test, dev_test, meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(test, test + meansSize);
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
		cudaFree(dev_assignedClusters);
	}

	end = clock();
	std::cout << "Time required for execution: "
		<< (double)(end - start) / CLOCKS_PER_SEC
		<< " seconds." << "\n\n";

	return cudaStatus;
}