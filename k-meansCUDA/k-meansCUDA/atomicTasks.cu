#include "atomicTasks.cuh"
#include "atomicKernels.cuh"
#include "helpers.h"

#include <time.h>
#include <stdio.h>
#include <iostream>


cudaError_t countKMeansAtomic(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
	value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
	uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
	cudaError_t cudaStatus;

	// Launch a kernel on the GPU with one thread for each element.
	int blockSizeN = BLOCK_SIZE;
	int nBlocksN = (dataSize - 1) / blockSizeN + 1;

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
		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterAtomicKernel << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dimension, 0, dataSize);
			synchronizeDevice();
			countDivMeansKernel << <meansBlocks, meansPerBlock * dimension >> >(meansSize, dev_counts, dev_means, dev_meansSums, dimension, meansPerBlock);
			synchronizeDevice();

			cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
			cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
		}

		// Check for any errors launching the kernel
		checkErrors();

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		synchronizeDevice();

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
	cudaFree(dev_meansSums);
	cudaFree(dev_data);
	cudaFree(dev_assignedClusters);
	cudaFree(dev_counts);

	end = clock();
	std::cout << "Time required for execution: "
		<< (double)(end - start) / CLOCKS_PER_SEC
		<< " seconds." << "\n\n";

	return cudaStatus;
}

cudaError_t countKMeansBIGDataAtomic(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
	value_t* dev_means = 0, *dev_data1 = 0, *dev_data2 = 0, *dev_meansSums = 0, *dev_temp = 0;
	uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
	cudaError_t cudaStatus;
	uint32_t dataPartSize, dataPartsCount, availableDataMem;
	size_t freeMem, totMem;

	uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
	//value_t* testMeans = (value_t*)calloc(meansSize * dimension , sizeof(value_t));

	// Launch a kernel on the GPU with one thread for each element.
	int blockSizeN = BLOCK_SIZE;
	//int blockSizeM = 16;
	//int nBlocksM = (meansSize - 1) / blockSizeM + 1;

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

		cudaMemGetInfo(&freeMem, &totMem);
		availableDataMem = totMem - sizeof(value_t)* dimension * meansSize * 2 - sizeof(uint32_t)* meansSize;
		if (availableDataMem < dataSize * dimension * sizeof(value_t))
		{
			// data size / (half of availavle memory)
			dataPartsCount = (uint32_t)ceil((sizeof(value_t)* dimension * dataSize) / (0.5 * availableDataMem));
			dataPartSize = (uint32_t)ceil(dataSize / (float)dataPartsCount);
		}
		else
		{
			dataPartsCount = 1;
			dataPartSize = dataSize;
		}

		int nBlocksN = (dataPartSize - 1) / blockSizeN + 1;

		// Allocate GPU buffers for three vectors (two input, one output)    .
		allocateMemory((void**)&dev_means, meansSize * dimension * sizeof(value_t));

		allocateAndSetMemory((void**)&dev_meansSums, meansSize * dimension * sizeof(value_t), 0);

		allocateMemory((void**)&dev_data1, dataPartSize * dimension * sizeof(value_t));
		allocateMemory((void**)&dev_data2, dataPartSize * dimension * sizeof(value_t));

		//allocateMemory((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));

		allocateAndSetMemory((void**)&dev_counts, meansSize * sizeof(uint32_t), 0);

		// Copy input vectors from host memory to GPU buffers.
		copyMemory(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
		copyMemory(dev_data1, data, dataPartSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);

		for (uint32_t i = 0; i < iterations; ++i)
		{
			for (size_t j = 0; j < dataPartsCount; ++j)
			{
				cudaMemcpyAsync(dev_data2, data + ((j + 1) % dataPartsCount) * dataPartSize * dimension, dataPartSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
				findNearestClusterAtomicKernel << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dev_meansSums, dataPartSize, dev_data1, dev_counts, dimension, dataPartsCount * j, dataSize);
				synchronizeDevice();
				std::swap(dev_data1, dev_data2);
			}
			//cudaMemcpy(test, dev_counts, sizeof(uint32_t)* meansSize, cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(test, test + meansSize);
			//uint32_t cSum = 0;
			//for (size_t i = 0; i < t.size(); i++)
			//{
			//	cSum += t[i];
			//}

			//cudaStatus = cudaGetLastError();

			//TO-DO ZLEPSIT! (jako atomicDivMeansKernel)
			countDivMeansKernel << <meansBlocks, meansPerBlock * dimension >> >(meansSize, dev_counts, dev_means, dev_meansSums, dimension, meansPerBlock);
			synchronizeDevice();

			cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
			cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
		}

		// Check for any errors launching the kernel
		checkErrors();

		copyMemory(means, dev_means, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);

		//copyMemory(assignedClusters, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
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
	cudaFree(dev_data1);
	cudaFree(dev_data2);
	cudaFree(dev_means);
	cudaFree(dev_meansSums);
	//cudaFree(dev_assignedClusters);
	cudaFree(dev_counts);

	end = clock();
	std::cout << "Time required for execution: "
		<< (double)(end - start) / CLOCKS_PER_SEC
		<< " seconds." << "\n\n";

	return cudaStatus;
}
