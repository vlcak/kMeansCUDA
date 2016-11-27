//#include "baseKernel.h"
#include "simpleKernels.cuh"
#include "atomicKernels.cuh"
#include "manyDimensionsKernels.cuh"
#include "fewMeansKernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "driver_types.h"

#include <stdint.h>
#include <stdexcept>
#include <stdio.h> 
#include <sstream>
#include <iostream>

#include <time.h>

#include <stdlib.h>
#include <vector>

uint64_t dimension;
typedef float value_t;
typedef unsigned char cluster_t;

const int BLOCK_SIZE = 64;
const int WARP_SIZE = 32;

#ifdef __CUDACC__
#pragma message "using nvcc"
#ifdef __CUDA_ARCH__
#pragma message "device code trajectory"
#if __CUDA_ARCH__ < 300
#pragma message "compiling for Fermi and older"
#elif __CUDA_ARCH__ < 500
#pragma message "compiling for Kepler"
#else
#pragma message "compiling for Maxwell"
#endif
#endif
#else
#pragma message "non - nvcc code trajectory"
#endif


cudaError_t countKMeansSimple(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansAtomic(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansBIGDataAtomic(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansManyDims(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansFewMeans(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansFewMeansV2(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansFewMeansV3(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);

#pragma region Kernels
#pragma endregion

#pragma region Common Functions
void usage()
{
    std::cout << "Usage:" << std::endl << "kmeans <data_file> <means_file> <clusters_file> <k> <iterations>" << std::endl << "kmeans --generate <data_file> <size> <seed>" << std::endl;
}

value_t* load(const std::string& file_name, uint64_t& dataSize)
{
	FILE* f = fopen(file_name.c_str(), "rb");
	if (!f) throw std::runtime_error("cannot open file for reading");
	//if (fseek(f, 0, SEEK_END)) throw std::runtime_error("seeking failed");
	if (!fread(&dataSize, sizeof(uint64_t), 1, f))  throw std::runtime_error("size cannot be read");
	if (!fread(&dimension, sizeof(uint64_t), 1, f))  throw std::runtime_error("dimension cannot be read");
	value_t* data = (value_t*)calloc(dataSize * dimension, sizeof(value_t));
	if (!fread(data, sizeof(value_t), dataSize * dimension, f))  throw std::runtime_error("value cannot be read");
	return data;
}

template<typename T>
T lexical_cast(const std::string& x)
{
	std::istringstream stream(x);
	T res;
	stream >> res;
	return res;
}

void save_results(const std::string& means_file_name, const std::string& clusters_file_name, const uint32_t meansSize, const value_t* means, const uint32_t dataSize, const value_t* data, const uint32_t* assignedClusters)
{
	FILE* f = fopen(means_file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	if (!fwrite(&dimension, sizeof(uint64_t), 1, f)) throw std::runtime_error("dimension cannot be written");
	//if (!fwrite(means, sizeof(value_t), dimension * meansSize, f)) throw std::runtime_error("value cannot be written");
	for (size_t i = 0; i < meansSize; i++)
	{
		if (!fwrite(&means[i*dimension], sizeof(value_t), dimension, f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&i, sizeof(unsigned char), 1, f)) throw std::runtime_error("value cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");


	f = fopen(clusters_file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	if (!fwrite(&dimension, sizeof(uint64_t), 1, f)) throw std::runtime_error("dimension cannot be written");
	for (size_t i = 0; i < dataSize; i++)
	{
		if (!fwrite(&data[i*dimension], sizeof(value_t), dimension, f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&assignedClusters[i], sizeof(unsigned char), 1, f)) throw std::runtime_error("value cannot be written");
		//if (!fwrite(&i, sizeof(value_t), 1, f)) throw std::runtime_error("distance cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");
}

int main(int argc, const char* argv[])
{
	if (argc == 6)
	{
		std::string file_name(argv[1]);
		std::string means_file_name = file_name;//(argv[2]);
		std::string clusters_file_name = file_name;// (argv[3]);
		int dataPos = file_name.find_last_of("/") + 1;
		means_file_name.erase(0, dataPos + std::string("data").length());
		means_file_name.insert(0, "means");
		clusters_file_name.erase(0, dataPos + std::string("data").length());
		clusters_file_name.insert(0, "cluster");
		std::string s_k(argv[4]);
		std::string s_iterations(argv[5]);
		uint32_t k = lexical_cast<uint32_t>(s_k);
		uint32_t iterations = lexical_cast<uint32_t>(s_iterations);
		uint64_t dataSize;

		value_t* data = load(file_name, dataSize);
		value_t* means = (value_t*)calloc(k * dimension, sizeof(value_t));
		uint32_t* assignedClusters = (uint32_t*)calloc(dataSize * dimension, sizeof(uint32_t));
		memcpy(means, data, k * dimension * sizeof(value_t));

		// Add vectors in parallel.
		cudaError_t cudaStatus = countKMeansSimple(iterations, dataSize, data, k, means, assignedClusters, dimension);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		save_results(means_file_name, clusters_file_name, k, means, dataSize, data, assignedClusters);

		free(data);
		free(means);
		free(assignedClusters);

		return 0;
	}
	usage();
	return 1;
}

#pragma endregion

#pragma region Tasks
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
			findNearestClusterAtomicKernel << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dimension, 0, dataSize);
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
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			throw 1;
		}
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

		// Allocate GPU buffers for means and means sumes.
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

		// Allocate GPU buffers for data.
		cudaStatus = cudaMalloc((void**)&dev_data1, dataPartSize * dimension * sizeof(value_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}
		// Allocate GPU buffers for data.
		cudaStatus = cudaMalloc((void**)&dev_data2, dataPartSize * dimension * sizeof(value_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}

		//cudaStatus = cudaMalloc((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));
		//if (cudaStatus != cudaSuccess) {
		//    fprintf(stderr, "cudaMalloc failed!");
		//    throw 1;
		//}

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

		cudaStatus = cudaMemcpy(dev_data1, data, dataPartSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			throw 1;
		}
		for (uint32_t i = 0; i < iterations; ++i)
		{
			for (size_t j = 0; j < dataPartsCount; ++j)
			{
				cudaMemcpyAsync(dev_data2, data + ((j + 1) % dataPartsCount) * dataPartSize * dimension, dataPartSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
				findNearestClusterAtomicKernel << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dev_meansSums, dataPartSize, dev_data1, dev_counts, dimension, dataPartsCount * j, dataSize);
				cudaDeviceSynchronize();
				cudaStatus = cudaGetLastError();
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
			cudaDeviceSynchronize();
			//cudaStatus = cudaGetLastError();

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
		cudaFree(dev_data1);
		cudaFree(dev_data2);
		cudaFree(dev_means);
		cudaFree(dev_meansSums);
		//cudaFree(dev_assignedClusters);
		cudaFree(dev_counts);
	}

    end = clock();
    std::cout << "Time required for execution: "
        << (double)(end - start) / CLOCKS_PER_SEC
        << " seconds." << "\n\n";

    return cudaStatus;
}

cudaError_t countKMeansManyDims(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
	value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
	uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
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

		// Launch a kernel on the GPU with one thread for each element.
		//int nBlocksN = (dataSize - 1) / blockSizeN + 1;
		//int blockSizeM = 16;
		//int nBlocksM = (meansSize - 1) / blockSizeM + 1;
		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterManyDimUnrolledKernel<< <nBlocksN, blockGrid, sizeof(value_t) * blockSizeN >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dev_assignedClusters, dimension);
			//findNearestClusterManyDimShuffleKernel << <nBlocksN, blockGrid>> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dev_assignedClusters, dimension);
			cudaDeviceSynchronize();
			//cudaMemcpy(testAssigned, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(testAssigned, testAssigned + dataSize);
			//cudaMemcpy(testDistances, dev_meansSums, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
			//std::vector<value_t> t2(testDistances, testDistances + meansSize);
			//cudaMemcpy(testCounts, dev_counts, meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t3(testCounts, testCounts+ meansSize);

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

cudaError_t countKMeansFewMeans(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
    value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
    uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
    cudaError_t cudaStatus;

    uint32_t cellsCount = 4;


    uint32_t* testAssigned, *testCounts;
    value_t* testDistances;

    testAssigned = (uint32_t*)malloc(dataSize * sizeof(uint32_t));
    testCounts = (uint32_t*)malloc(cellsCount * meansSize * sizeof(uint32_t));
	testDistances = (value_t*)malloc(cellsCount * meansSize * dimension * sizeof(value_t));

	// Launch a kernel on the GPU with one thread for each element.
	int blockSizeN = BLOCK_SIZE;
	dim3 gridSize;

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

		cudaStatus = cudaMalloc((void**)&dev_meansSums, cellsCount * meansSize * dimension * sizeof(value_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}
		else
		{
			cudaMemset(dev_meansSums, 0, cellsCount * meansSize * dimension * sizeof(value_t));
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

		cudaStatus = cudaMalloc((void**)&dev_counts, cellsCount * meansSize * sizeof(uint32_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}
		else
		{
			cudaMemset(dev_counts, 0, cellsCount * meansSize * sizeof(uint32_t));
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

		gridSize.x = ((dataSize - 1) / blockSizeN) / cellsCount + 1;
		gridSize.y = cellsCount;
		//int blockSizeM = 16;
		//int nBlocksM = (meansSize - 1) / blockSizeM + 1;
		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterFewMeansKernel << <gridSize, blockSizeN >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dev_assignedClusters, dimension);
			cudaDeviceSynchronize();

			//cudaMemcpy(testAssigned, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(testAssigned, testAssigned + dataSize);
			//cudaMemcpy(testDistances, dev_meansSums, cellsCount * meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
			//std::vector<value_t> t2(testDistances, testDistances + meansSize * cellsCount);
			//cudaMemcpy(testCounts, dev_counts, cellsCount * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t3(testCounts, testCounts + meansSize * cellsCount);

			//std::vector<uint32_t> t5(t.begin() + 9900, t.end());

			//int sum_of_elems2 = 0;
			//for (size_t i = 0; i < 128; i++)
			//{
			//    sum_of_elems2 += t3[i];
			//}

			countDivFewMeansKernel << <meansSize, dimension >> >(meansSize, dev_counts, dev_means, dev_meansSums, dimension, cellsCount);
			cudaDeviceSynchronize();

			//cudaMemcpy(testCounts, dev_counts, cellsCount * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t4(testCounts, testCounts + meansSize * cellsCount);


			//int sum_of_elems = 0;
			//for (size_t i = 0; i < BLOCK_SIZE; i++)
			//{
			//    sum_of_elems += t4[i];
			//}

			cudaMemset(dev_meansSums, 0, cellsCount * meansSize * dimension * sizeof(value_t));
			cudaMemset(dev_counts, 0, cellsCount * meansSize * sizeof(uint32_t));
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

cudaError_t countKMeansFewMeansV2(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
    value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
    uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
    cudaError_t cudaStatus;

    uint32_t cellsCount = 4;


    uint32_t* testAssigned, *testCounts;
    value_t* testDistances;

    testAssigned = (uint32_t*)malloc(dataSize * sizeof(uint32_t));
    testCounts = (uint32_t*)malloc(cellsCount * meansSize * sizeof(uint32_t));
    testDistances = (value_t*)malloc(cellsCount * meansSize * dimension * sizeof(value_t));

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

		cudaStatus = cudaMalloc((void**)&dev_meansSums, dataSize * meansSize * dimension * sizeof(value_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}
		else
		{
			cudaMemset(dev_meansSums, 0, dataSize * meansSize * dimension * sizeof(value_t));
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

		cudaStatus = cudaMalloc((void**)&dev_counts, dataSize * meansSize * sizeof(uint32_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}
		else
		{
			cudaMemset(dev_counts, 0, dataSize * meansSize * sizeof(uint32_t));
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


		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterFewMeansKernelV2 << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dev_assignedClusters, dimension);
			cudaDeviceSynchronize();

			//cudaMemcpy(testAssigned, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(testAssigned, testAssigned + dataSize);
			//cudaMemcpy(testDistances, dev_meansSums, cellsCount * meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
			//std::vector<value_t> t2(testDistances, testDistances + meansSize * cellsCount);
			//cudaMemcpy(testCounts, dev_counts, cellsCount * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t3(testCounts, testCounts + meansSize * cellsCount);

			//std::vector<uint32_t> t5(t.begin() + 9900, t.end());

			//int sum_of_elems2 = 0;
			//for (size_t i = 0; i < 128; i++)
			//{
			//    sum_of_elems2 += t3[i];
			//}

			countDivFewMeansKernelV2 << <meansSize, dimension >> >(dataSize, meansSize, dev_counts, dev_means, dev_meansSums, dimension, cellsCount);
			cudaDeviceSynchronize();

			//cudaMemcpy(testCounts, dev_counts, cellsCount * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t4(testCounts, testCounts + meansSize * cellsCount);


			//int sum_of_elems = 0;
			//for (size_t i = 0; i < BLOCK_SIZE; i++)
			//{
			//    sum_of_elems += t4[i];
			//}

			cudaMemset(dev_meansSums, 0, dataSize * meansSize * dimension * sizeof(value_t));
			cudaMemset(dev_counts, 0, dataSize * meansSize * sizeof(uint32_t));
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

cudaError_t countKMeansFewMeansV3(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
    value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
    uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
    cudaError_t cudaStatus;
    int blockSizeN = BLOCK_SIZE;
    int nBlocksN = (dataSize - 1) / blockSizeN + 1;

    uint32_t* testAssigned, *testCounts;
    value_t* testDistances;

    testAssigned = (uint32_t*)malloc(dataSize * sizeof(uint32_t));
    testCounts = (uint32_t*)malloc(nBlocksN * meansSize * sizeof(uint32_t));
    testDistances = (value_t*)malloc(nBlocksN * meansSize * dimension * sizeof(value_t));
	
	uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
	value_t* testMeans = (value_t*)calloc(meansSize * dimension, sizeof(value_t));


	// Launch a kernel on the GPU with one thread for each element.
	//dim3 gridSize;
	//gridSize.y = cellsCount;
	//int blockSizeM = 16;
	//int nBlocksM = (meansSize - 1) / blockSizeM + 1;
	int sharedMemorySize = (sizeof(value_t)* dimension + sizeof(uint8_t)) * meansSize * blockSizeN;
	cudaDeviceProp prop;
	cudaFuncSetCacheConfig(findNearestClusterFewMeansKernelV3, cudaFuncCache::cudaFuncCachePreferShared);
	cudaGetDeviceProperties(&prop, 0);// cudaGetDeviceProp::sharedMemPerBlock;
	size_t i = prop.sharedMemPerMultiprocessor;
	uint32_t reductionThreadCount = (uint32_t)pow(2, floor(log2((float)nBlocksN)));

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

		cudaStatus = cudaMalloc((void**)&dev_meansSums, nBlocksN * meansSize * dimension * sizeof(value_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}
		else
		{
			cudaMemset(dev_meansSums, 0, nBlocksN * meansSize * dimension * sizeof(value_t));
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

		cudaStatus = cudaMalloc((void**)&dev_counts, nBlocksN * meansSize * sizeof(uint32_t));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			throw 1;
		}
		else
		{
			cudaMemset(dev_counts, 0, nBlocksN * meansSize * sizeof(uint32_t));
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

		for (uint32_t i = 0; i < iterations; ++i)
		{
			findNearestClusterFewMeansKernelV3 << <nBlocksN, blockSizeN, sharedMemorySize >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dev_assignedClusters, dimension);
			cudaDeviceSynchronize();

			//cudaMemcpy(testAssigned, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t(testAssigned, testAssigned + dataSize);
			//cudaMemcpy(testDistances, dev_meansSums, blockSizeN * meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
			//std::vector<value_t> t2(testDistances, testDistances + meansSize * blockSizeN);
			//cudaMemcpy(testCounts, dev_counts, nBlocksN * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t3(testCounts, testCounts + meansSize * nBlocksN);

			//std::vector<uint32_t> t5(t.begin() + 9900, t.end());

			//int sum_of_elems2 = 0;
			//for (size_t i = 0; i < meansSize * nBlocksN; i++)
			//{
			//    sum_of_elems2 += t3[i];
			//}

			countDivFewMeansKernelV3 << <1, reductionThreadCount >> >(dataSize, meansSize, dev_counts, dev_means, dev_meansSums, dimension, nBlocksN);
			cudaDeviceSynchronize();

			//cudaMemcpy(testCounts, dev_counts, nBlocksN * meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//std::vector<uint32_t> t4(testCounts, testCounts + meansSize);

			//int sum_of_elems = 0;
			//for (size_t i = 0; i < meansSize; i++)
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

#pragma endregion Tasks