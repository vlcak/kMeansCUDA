//#include "baseKernel.h"
//#include "simpleKernels.cu"
//#include "atomicKernels.cu"
//#include "manyDimensionsKernels.cu";

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>
#include <stdio.h> 
#include <sstream>
#include <iostream>

#include <time.h>

#include <stdlib.h>
#include <vector>

//#include "baseKernel.h"
uint64_t dimension;
typedef float value_t;
typedef unsigned char cluster_t;

cudaError_t countKMeansSimple(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansAtomic(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansManyDims(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);
cudaError_t countKMeansFewMeans(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension);

void usage()
{
	std::cout << "Usage:" << std::endl << "kmeans <data_file> <means_file> <clusters_file> <k> <iterations>" << std::endl << "kmeans --generate <data_file> <size> <seed>" << std::endl;
}

#pragma region Kernels
__global__ void findNearestClusterKernel(const uint32_t meansSize, const value_t *means, const uint32_t dataSize, const value_t* data, uint32_t* assignedClusters, const uint32_t dimension)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    value_t minDistance = LLONG_MAX, distance = 0, difference = 0;
    for (size_t i = 0; i < meansSize; ++i)
    {
        distance = 0;
        for (size_t j = 0; j < dimension; ++j)
        {
            difference = means[i * dimension + j] - data[id * dimension + j];
            distance += difference * difference;
        }
        if (minDistance > distance)
        {
            minDistance = distance;
            assignedClusters[id] = i;
        }
    }
}

__global__ void countNewMeansKernel(uint32_t* assignedClusters, const uint32_t dataSize, const value_t* data, value_t* means, const uint32_t dimension, uint32_t* test)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int idOffset = id * dimension;
    uint32_t count = 0;
    for (size_t i = idOffset; i < idOffset + dimension; ++i)
    {
        means[i] = 0;
    }
    for (size_t i = 0; i < dataSize; ++i)
    {
        if (assignedClusters[i] == id)
        {
            for (size_t j = 0; j < dimension; ++j)
            {
                means[idOffset + j] += data[i * dimension + j];
            }
            ++count;
        }
    }
    for (size_t i = idOffset; i < idOffset + dimension; ++i)
    {
        means[i] /= count;
    }
    test[id] = count;
}

__global__ void findNearestClusterAtomicKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    value_t minDistance = LLONG_MAX, distance = 0, difference = 0;
    int clusterID = -1;
    for (size_t i = 0; i < meansSize; ++i)
    {
        distance = 0;
        for (size_t j = 0; j < dimension; ++j)
        {
            difference = means[i * dimension + j] - data[id * dimension + j];
            distance += difference * difference;
        }
        if (minDistance > distance)
        {
            minDistance = distance;
            clusterID = i;
        }
    }
    atomicInc(&counts[clusterID], INT32_MAX);
    assignedClusters[id] = clusterID;
    for (size_t j = 0; j < dimension; ++j)
    {
        atomicAdd(&measnSums[clusterID * dimension + j], data[id * dimension + j]);
    }
}

__global__ void countDivMeansKernel(const uint32_t meansSize, const uint32_t* counts, value_t* means, const value_t* meansSums, const uint32_t dimension)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    means[id] = meansSums[id] / (value_t)counts[blockIdx.x];
}

__global__ void findNearestClusterManyDimKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters)
{
    //int id = threadIdx.x;
    value_t minDistance = LLONG_MAX, difference = 0;
    int clusterID = -1;
    extern __shared__ value_t distances[];

    for (size_t i = 0; i < meansSize; ++i)
    {
        difference = means[i * blockDim.x + threadIdx.x] - data[blockIdx.x * blockDim.x + threadIdx.x];
        distances[threadIdx.x] = difference * difference;
        //sum distances in block
        __syncthreads();
        for (size_t j = blockDim.x / 2; j > 0; j >>= 1)
        {
            if (threadIdx.x < j)
            {
                distances[threadIdx.x] += distances[threadIdx.x + j];
            }
            __syncthreads();
        }

        if ( (minDistance > distances[0]))
        {
            minDistance = distances[0];
            clusterID = i;
        }
    }

    if (threadIdx.x == 0)
    {
        atomicInc(&counts[clusterID], INT32_MAX);
        assignedClusters[blockIdx.x] = clusterID;
    }

    atomicAdd(&measnSums[clusterID * blockDim.x + threadIdx.x], data[blockIdx.x * blockDim.x + threadIdx.x]);
}

template <uint32_t blockSize>
__global__ void findNearestClusterManyDimUnrolledKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters)
{
    //int id = threadIdx.x;
    value_t minDistance = LLONG_MAX, difference = 0;
    int clusterID = -1;
    extern __shared__ value_t distances[];

    uint32_t j;

    for (size_t i = 0; i < meansSize; ++i)
    {
        difference = means[i * blockDim.x + threadIdx.x] - data[blockIdx.x * blockDim.x + threadIdx.x];
        distances[threadIdx.x] = difference * difference;
        //sum distances in block
        __syncthreads();


        if (blockSize >= 512) { if (threadIdx.x < 256) { distances[threadIdx.x] += distances[threadIdx.x + 256]; } __syncthreads(); }
        if (blockSize >= 256) { if (threadIdx.x < 128) { distances[threadIdx.x] += distances[threadIdx.x + 128]; } __syncthreads(); }
        if (blockSize >= 128) { if (threadIdx.x <  64) { distances[threadIdx.x] += distances[threadIdx.x +  64]; } __syncthreads(); }

        if (threadIdx.x < 32)
        {
            if (blockSize >= 64) distances[threadIdx.x] += distances[threadIdx.x + 32];
            if (blockSize >= 32) distances[threadIdx.x] += distances[threadIdx.x + 16];
            if (blockSize >= 16) distances[threadIdx.x] += distances[threadIdx.x + 8];
            if (blockSize >= 8) distances[threadIdx.x] += distances[threadIdx.x + 4];
            if (blockSize >= 4) distances[threadIdx.x] += distances[threadIdx.x + 2];
            if (blockSize >= 2) distances[threadIdx.x] += distances[threadIdx.x + 1];
        }


        if ((minDistance > distances[0]))
        {
            minDistance = distances[0];
            clusterID = i;
        }
    }

    if (threadIdx.x == 0)
    {
        atomicInc(&counts[clusterID], INT32_MAX);
        assignedClusters[blockIdx.x] = clusterID;
    }

    atomicAdd(&measnSums[clusterID * blockDim.x + threadIdx.x], data[blockIdx.x * blockDim.x + threadIdx.x]);
}

__global__ void findNearestClusterFewMeansKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension)
{
    unsigned int id = threadIdx.x + blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x);
    value_t minDistance = LLONG_MAX, distance = 0, difference = 0;
    int clusterID = -1;
    for (size_t i = 0; i < meansSize; ++i)
    {
        distance = 0;
        for (size_t j = 0; j < dimension; ++j)
        {
            difference = means[i * dimension + j] - data[id * dimension + j];
            distance += difference * difference;
        }
        if (minDistance > distance)
        {
            minDistance = distance;
            clusterID = i;
        }
    }
    if (clusterID != -1)
    {
        atomicInc(&counts[blockIdx.y * meansSize + clusterID], INT32_MAX);
        assignedClusters[id] = clusterID;
        for (size_t j = 0; j < dimension; ++j)
        {
            atomicAdd(&measnSums[blockIdx.y * meansSize * dimension + clusterID * dimension + j], data[id * dimension + j]);
        }
    }
}

__global__ void countDivFewMeansKernel(const uint32_t meansSize, uint32_t* counts, value_t* means, const value_t* meansSums, const uint32_t dimension, const uint32_t cellsCount)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    uint32_t count = 0;

    means[id] = meansSums[id];

    count = counts[blockIdx.x];

    for (size_t i = 1; i < cellsCount; i++)
    {
        means[id] += meansSums[i * dimension * meansSize + id];
        count += counts[i * meansSize + blockIdx.x];
    }

    means[id] /= count;

    if (threadIdx.x == 0)
    {
        counts[blockIdx.x] = count;
    }
}
#pragma endregion

#pragma region Common Functions
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
		std::string means_file_name(argv[2]);
		std::string clusters_file_name(argv[3]);
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
        cudaError_t cudaStatus = countKMeansFewMeans(iterations, dataSize, data, k, means, assignedClusters, dimension);
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

    clock_t start, end;
    start = clock();

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_means, meansSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_data, dataSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_test, meansSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));

    // Launch a kernel on the GPU with one thread for each element.
    int blockSizeN = 32;
    int nBlocksN = (dataSize - 1) / blockSizeN + 1;
    int blockSizeM = 16;
    int nBlocksM = (meansSize - 1) / blockSizeM + 1;
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
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(means, dev_means, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(assignedClusters, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



Error:
    cudaFree(dev_data);
    cudaFree(dev_means);
    cudaFree(dev_assignedClusters);

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

    clock_t start, end;
    start = clock();

    //std::vector<uint32_t> testVector(meansSize);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_means, meansSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_meansSums, meansSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    else
    {
        cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
    }

    cudaStatus = cudaMalloc((void**)&dev_data, dataSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_counts, meansSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    else
    {
        cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
    //value_t* testMeans = (value_t*)calloc(meansSize * dimension , sizeof(value_t));

    // Launch a kernel on the GPU with one thread for each element.
    int blockSizeN = 32;
    int nBlocksN = (dataSize - 1) / blockSizeN + 1;
    //int blockSizeM = 16;
    //int nBlocksM = (meansSize - 1) / blockSizeM + 1;
    for (uint32_t i = 0; i < iterations; ++i)
    {
        findNearestClusterAtomicKernel << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dev_assignedClusters, dimension);
        cudaDeviceSynchronize();
        countDivMeansKernel << <meansSize, dimension >> >(meansSize, dev_counts, dev_means, dev_meansSums, dimension);
        cudaDeviceSynchronize();

        cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
        cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(means, dev_means, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(assignedClusters, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



Error:
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

cudaError_t countKMeansManyDims(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
    value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
    uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
    cudaError_t cudaStatus;

    //uint32_t* testAssigned,* testCounts;
    //value_t* testDistances;

    //testAssigned = (uint32_t*)malloc(dataSize * sizeof(uint32_t));
    //testCounts = (uint32_t*)malloc(meansSize * sizeof(uint32_t));
    //testDistances = (value_t*)malloc(meansSize * dimension * sizeof(value_t));

    clock_t start, end;
    start = clock();

    //std::vector<uint32_t> testVector(meansSize);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_means, meansSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_meansSums, meansSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    else
    {
        cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
    }

    cudaStatus = cudaMalloc((void**)&dev_data, dataSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_counts, meansSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    else
    {
        cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
    //value_t* testMeans = (value_t*)calloc(meansSize * dimension , sizeof(value_t));

    // Launch a kernel on the GPU with one thread for each element.
    int blockSizeN = 32;
    int nBlocksN = (dataSize - 1) / blockSizeN + 1;
    //int blockSizeM = 16;
    //int nBlocksM = (meansSize - 1) / blockSizeM + 1;
    for (uint32_t i = 0; i < iterations; ++i)
    {
        findNearestClusterManyDimUnrolledKernel<32><< <dataSize, dimension, sizeof(value_t)* dimension >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dev_assignedClusters);
        cudaDeviceSynchronize();
        //cudaMemcpy(testAssigned, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //std::vector<uint32_t> t(testAssigned, testAssigned + dataSize);
        //cudaMemcpy(testDistances, dev_meansSums, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
        //std::vector<value_t> t2(testDistances, testDistances + meansSize);
        //cudaMemcpy(testCounts, dev_counts, meansSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        //std::vector<uint32_t> t3(testCounts, testCounts+ meansSize);
        countDivMeansKernel << <meansSize, dimension >> >(meansSize, dev_counts, dev_means, dev_meansSums, dimension);
        cudaDeviceSynchronize();

        cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
        cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(means, dev_means, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(assignedClusters, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



Error:
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

cudaError_t countKMeansFewMeans(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
    value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
    uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
    cudaError_t cudaStatus;

    uint32_t cellsCount = 4;


    uint32_t* testAssigned,* testCounts;
    value_t* testDistances;

    testAssigned = (uint32_t*)malloc(dataSize * sizeof(uint32_t));
    testCounts = (uint32_t*)malloc(cellsCount * meansSize * sizeof(uint32_t));
    testDistances = (value_t*)malloc(cellsCount * meansSize * dimension * sizeof(value_t));

    clock_t start, end;
    start = clock();

    //std::vector<uint32_t> testVector(meansSize);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_means, meansSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_meansSums, cellsCount * meansSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    else
    {
        cudaMemset(dev_meansSums, 0, cellsCount * meansSize * dimension * sizeof(value_t));
    }

    cudaStatus = cudaMalloc((void**)&dev_data, dataSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_counts, cellsCount * meansSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    else
    {
        cudaMemset(dev_counts, 0, cellsCount * meansSize * sizeof(uint32_t));
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
    //value_t* testMeans = (value_t*)calloc(meansSize * dimension , sizeof(value_t));

    // Launch a kernel on the GPU with one thread for each element.
    int blockSizeN = 32;
    dim3 gridSize;
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
        //for (size_t i = 0; i < 32; i++)
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
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(means, dev_means, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(assignedClusters, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



Error:
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

#pragma endregion Tasks