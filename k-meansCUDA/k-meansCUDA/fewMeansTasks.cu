#include "fewMeansTasks.cuh"
#include "fewMeansKernels.cuh"
#include "helpers.h"

#include <time.h>
#include <stdio.h>
#include <iostream>

/// each thread computes one point, but blocks are in two dimensional grid and the y-th coordinate determinates which copy of means is used
cudaError_t countKMeansFewMeans(const uint32_t iterations, const uint32_t dataSize_u32, const value_t* data, const uint32_t meansSize_u32, value_t* means, uint32_t* assignedClusters, uint64_t dimension_u64, uint32_t cellsCount)
{
	value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
	uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
	const my_size_t dataSize = static_cast<my_size_t>(dataSize_u32);
	const my_size_t meansSize = static_cast<my_size_t>(meansSize_u32);
	const my_size_t dimension = static_cast<my_size_t>(dimension_u64);
	cudaError_t cudaStatus;

	//uint32_t* testAssigned, *testCounts;
	//value_t* testDistances;

	//testAssigned = (uint32_t*)malloc(dataSize * sizeof(uint32_t));
	//testCounts = (uint32_t*)malloc(cellsCount * meansSize * sizeof(uint32_t));
	//testDistances = (value_t*)malloc(cellsCount * meansSize * dimension * sizeof(value_t));

	// Launch a kernel on the GPU with one thread for each element.
	dim3 blockSizeN;
	blockSizeN.x = BLOCK_SIZE / cellsCount;
	blockSizeN.y = cellsCount;
	int gridSize = ((dataSize - 1) / BLOCK_SIZE) + 1;
	//gridSize.x = ((dataSize - 1) / BLOCK_SIZE) / cellsCount  + 1;
	//gridSize.y = cellsCount;

	dim3 blockSizeMeans;
	blockSizeMeans.x = dimension;
	blockSizeMeans.y = BLOCK_SIZE / dimension;

	clock_t start, end;
	start = clock();

	//std::vector<uint32_t> testVector(meansSize);
	try
	{
		//choose which GPU to run on, change this on a multi - GPU system.
		setDevice(DEVICE_ID);

		// Allocate GPU buffers for three vectors (two input, one output)    .
		allocateMemory((void**)&dev_means, meansSize * dimension * sizeof(value_t));

		allocateMemory((void**)&dev_meansSums, cellsCount * meansSize * dimension * sizeof(value_t));
		cudaMemset(dev_meansSums, 0, cellsCount * meansSize * dimension * sizeof(value_t));

		allocateMemory((void**)&dev_data, dataSize * dimension * sizeof(value_t));

		allocateMemory((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));

		allocateMemory((void**)&dev_counts, cellsCount * meansSize * sizeof(uint32_t));
		cudaMemset(dev_counts, 0, cellsCount * meansSize * sizeof(uint32_t));

		// Copy input vectors from host memory to GPU buffers.
		copyMemory(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);

		copyMemory(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);

		//uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
		//value_t* testMeans = (value_t*)calloc(meansSize * dimension , sizeof(value_t));

		//int blockSizeM = 16;
		//int nBlocksM = (meansSize - 1) / blockSizeM + 1;
		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterFewMeansKernel << <gridSize, blockSizeN >> >(meansSize, dev_means, dev_meansSums, dev_data, dev_counts, dev_assignedClusters, dimension);
			synchronizeDevice();

			//cudaMemcpy(testAssigned, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(testAssigned, testAssigned + dataSize);
			//cudaMemcpy(testDistances, dev_meansSums, cellsCount * meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
			//std::vector<value_t> t2(testDistances, testDistances + meansSize * cellsCount);
			//cudaMemcpy(testCounts, dev_counts, cellsCount * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t3(testCounts, testCounts + meansSize * cellsCount);

			//std::vector<uint32_t> t5(t.begin() + 9900, t.end());

			//int sum_of_elems2 = 0;
			//for (int32_t i = 0; i < 128; i++)
			//{
			//    sum_of_elems2 += t3[i];
			//}

			countDivFewMeansKernel << <meansSize, blockSizeMeans >> >(meansSize, dev_counts, dev_means, dev_meansSums, dimension, cellsCount);
			synchronizeDevice();

			//cudaMemcpy(testCounts, dev_counts, cellsCount * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t4(testCounts, testCounts + meansSize * cellsCount);


			//int sum_of_elems = 0;
			//for (int32_t i = 0; i < BLOCK_SIZE; i++)
			//{
			//    sum_of_elems += t4[i];
			//}

			cudaMemset(dev_meansSums, 0, cellsCount * meansSize * dimension * sizeof(value_t));
			cudaMemset(dev_counts, 0, cellsCount * meansSize * sizeof(uint32_t));
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

cudaError_t countKMeansFewMeansV2(const uint32_t iterations, const uint32_t dataSize_u32, const value_t* data, const uint32_t meansSize_u32, value_t* means, uint32_t* assignedClusters, uint64_t dimension_u64)
{
	value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
	uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
	const my_size_t dataSize = static_cast<my_size_t>(dataSize_u32);
	const my_size_t meansSize = static_cast<my_size_t>(meansSize_u32);
	const my_size_t dimension = static_cast<my_size_t>(dimension_u64);
	cudaError_t cudaStatus;

	//uint32_t cellsCount = 4;


	uint32_t* testAssigned, *testCounts;
	value_t* testDistances;

	testAssigned = (uint32_t*)malloc(dataSize * sizeof(uint32_t));
	//testCounts = (uint32_t*)malloc(cellsCount * meansSize * sizeof(uint32_t));
	//testDistances = (value_t*)malloc(cellsCount * meansSize * dimension * sizeof(value_t));

	// Launch a kernel on the GPU with one thread for each element.
	int blockSizeN = BLOCK_SIZE;
	//dim3 gridSize;
	int nBlocksN = (dataSize - 1) / blockSizeN + 1;
	//gridSize.y = cellsCount;
	//int blockSizeM = 16;
	//int nBlocksM = (meansSize - 1) / blockSizeM + 1;

	clock_t start, end;
	start = clock();

	//std::vector<uint32_t> testVector(meansSize);
	try
	{
		//choose which GPU to run on, change this on a multi - GPU system.
		setDevice(DEVICE_ID);

		// Allocate GPU buffers for three vectors (two input, one output)    .
		allocateMemory((void**)&dev_means, meansSize * dimension * sizeof(value_t));

		allocateAndSetMemory((void**)&dev_meansSums, dataSize * meansSize * dimension * sizeof(value_t), 0);

		allocateMemory((void**)&dev_data, dataSize * dimension * sizeof(value_t));

		allocateMemory((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));

		allocateAndSetMemory((void**)&dev_counts, dataSize * meansSize * sizeof(uint32_t), 0);

		// Copy input vectors from host memory to GPU buffers.
		copyMemory(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);

		copyMemory(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);

		//uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
		//value_t* testMeans = (value_t*)calloc(meansSize * dimension , sizeof(value_t));


		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterFewMeansKernelV2 << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dev_meansSums, dev_data, dev_counts, dev_assignedClusters, dimension);
			synchronizeDevice();

			//cudaMemcpy(testAssigned, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(testAssigned, testAssigned + dataSize);
			//cudaMemcpy(testDistances, dev_meansSums, cellsCount * meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
			//std::vector<value_t> t2(testDistances, testDistances + meansSize * cellsCount);
			//cudaMemcpy(testCounts, dev_counts, cellsCount * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t3(testCounts, testCounts + meansSize * cellsCount);

			//std::vector<uint32_t> t5(t.begin() + 9900, t.end());

			//int sum_of_elems2 = 0;
			//for (int32_t i = 0; i < 128; i++)
			//{
			//    sum_of_elems2 += t3[i];
			//}

			countDivFewMeansKernelV2 << <meansSize, dimension >> >(dataSize, meansSize, dev_counts, dev_means, dev_meansSums, dimension);
			synchronizeDevice();

			//cudaMemcpy(testCounts, dev_counts, cellsCount * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t4(testCounts, testCounts + meansSize * cellsCount);


			//int sum_of_elems = 0;
			//for (int32_t i = 0; i < BLOCK_SIZE; i++)
			//{
			//    sum_of_elems += t4[i];
			//}

			cudaMemset(dev_meansSums, 0, dataSize * meansSize * dimension * sizeof(value_t));
			cudaMemset(dev_counts, 0, dataSize * meansSize * sizeof(uint32_t));
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

cudaError_t countKMeansFewMeansV3(const uint32_t iterations, const uint32_t dataSize_u32, const value_t* data, const uint32_t meansSize_u32, value_t* means, uint32_t* assignedClusters, uint64_t dimension_u64, uint32_t sharedRedundancy, uint32_t globalRedundancy)
{
	value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
	uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
	const my_size_t dataSize = static_cast<my_size_t>(dataSize_u32);
	const my_size_t meansSize = static_cast<my_size_t>(meansSize_u32);
	const my_size_t dimension = static_cast<my_size_t>(dimension_u64);
	cudaError_t cudaStatus;

	dim3 blockSizeN(BLOCK_SIZE / sharedRedundancy, sharedRedundancy);
	dim3 nBlocksN = (((dataSize - 1) / BLOCK_SIZE / globalRedundancy) + 1, globalRedundancy);

	//uint32_t* testAssigned, *testCounts;
	//value_t* testDistances;

	//testAssigned = (uint32_t*)malloc(dataSize * sizeof(uint32_t));
	//testCounts = (uint32_t*)malloc(nBlocksN * meansSize * sizeof(uint32_t));
	//testDistances = (value_t*)malloc(nBlocksN * meansSize * dimension * sizeof(value_t));

	//uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
	//value_t* testMeans = (value_t*)calloc(meansSize * dimension, sizeof(value_t));


	// Launch a kernel on the GPU with one thread for each element.
	//dim3 gridSize;
	//gridSize.y = cellsCount;
	//int blockSizeM = 16;
	//int nBlocksM = (meansSize - 1) / blockSizeM + 1;
	int sharedMemorySize = (sizeof(value_t)* dimension + sizeof(uint32_t)) * meansSize * sharedRedundancy;
	cudaDeviceProp prop;
	cudaFuncSetCacheConfig(findNearestClusterFewMeansKernelV3, cudaFuncCache::cudaFuncCachePreferShared);
	cudaGetDeviceProperties(&prop, 0);// cudaGetDeviceProp::sharedMemPerBlock;
	size_t i = prop.sharedMemPerMultiprocessor;
	//uint32_t reductionThreadCount = (uint32_t)pow(2, floor(log2((float)globalRedundancy)));

	dim3 reductionThreadBlock = (dimension, globalRedundancy);
	reductionThreadBlock.z = BLOCK_SIZE / (reductionThreadBlock.x * reductionThreadBlock.y);
	int reductionBlocksN = (meansSize - 1) / reductionThreadBlock.z + 1;

	clock_t start, end;
	start = clock();

	//std::vector<uint32_t> testVector(meansSize);

	try
	{
		//choose which GPU to run on, change this on a multi - GPU system.
		setDevice(DEVICE_ID);

		// Allocate GPU buffers for three vectors (two input, one output)    .
		allocateMemory((void**)&dev_means, meansSize * dimension * sizeof(value_t));

		allocateMemory((void**)&dev_meansSums, globalRedundancy * meansSize * dimension * sizeof(value_t));

		allocateMemory((void**)&dev_data, dataSize * dimension * sizeof(value_t));

		allocateMemory((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));

		allocateMemory((void**)&dev_counts, globalRedundancy * meansSize * sizeof(uint32_t));

		// Copy input vectors from host memory to GPU buffers.
		copyMemory(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);

		copyMemory(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);

		for (uint32_t i = 0; i < iterations; ++i)
		{
			cudaMemset(dev_counts, 0, globalRedundancy * meansSize * sizeof(uint32_t));
			cudaMemset(dev_meansSums, 0, globalRedundancy * meansSize * dimension * sizeof(value_t));

			findNearestClusterFewMeansKernelV3 << <nBlocksN, blockSizeN, sharedMemorySize >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dev_assignedClusters, dimension);
			synchronizeDevice();

			//cudaMemcpy(testAssigned, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(testAssigned, testAssigned + dataSize);
			//cudaMemcpy(testDistances, dev_meansSums, blockSizeN * meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
			//std::vector<value_t> t2(testDistances, testDistances + meansSize * blockSizeN);
			//cudaMemcpy(testCounts, dev_counts, nBlocksN * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t3(testCounts, testCounts + meansSize * nBlocksN);

			//std::vector<uint32_t> t5(t.begin() + 9900, t.end());

			//int sum_of_elems2 = 0;
			//for (int32_t i = 0; i < meansSize * nBlocksN; i++)
			//{
			//    sum_of_elems2 += t3[i];
			//}

			countDivFewMeansKernelV3 << <reductionBlocksN, reductionThreadBlock >> >(meansSize, dev_counts, dev_means, dev_meansSums, dimension);
			synchronizeDevice();

			// reset means sums and counts

			//cudaMemcpy(testCounts, dev_counts, nBlocksN * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t4(testCounts, testCounts + meansSize);

			//int sum_of_elems = 0;
			//for (int32_t i = 0; i < meansSize; i++)
			//{
			//    sum_of_elems += t4[i];
			//}
			//cudaMemcpy(testDistances, dev_means, dimension * meansSize * sizeof(value_t), cudaMemcpyDeviceToHost);
			//std::vector<value_t> t6(testDistances, testDistances + meansSize * dimension);
			//cudaMemcpy(testDistances, dev_meansSums, dimension * meansSize * sizeof(value_t), cudaMemcpyDeviceToHost);
			//std::vector<value_t> t7(testDistances, testDistances + meansSize * dimension);

			//cudaMemset(dev_meansSums, 0, nBlocksN * meansSize * dimension * sizeof(value_t));
			//cudaMemset(dev_counts, 0, nBlocksN * meansSize * sizeof(uint32_t));
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
