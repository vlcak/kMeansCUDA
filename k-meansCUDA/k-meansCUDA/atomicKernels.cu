#include "atomicKernels.cuh"


__global__ void findNearestClusterAtomicKernel(const my_size_t meansSize, const value_t *means, value_t *measnSums, const my_size_t dataSize, const value_t* data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	value_t minDistance = LLONG_MAX, distance = 0, difference = 0;
	int clusterID = -1;
	for (my_size_t i = 0; i < meansSize; ++i)
	{
		distance = 0;
		for (my_size_t j = 0; j < dimension; ++j)
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
	if (id + dataOffset < totalDataSize)
	{
		atomicInc(&counts[clusterID], INT_MAX);
		//assignedClusters[id] = clusterID;
		for (my_size_t j = 0; j < dimension; ++j)
		{
			atomicAdd(&measnSums[clusterID * dimension + j], data[id * dimension + j]);
		}
	}
}

__global__ void findNearestClusterAtomicKernelTransposed(const my_size_t meansSize, const value_t *means, value_t *measnSums, const my_size_t dataSize, const value_t* data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	value_t minDistance = LLONG_MAX, distance = 0, difference = 0;
	int clusterID = -1;
	for (my_size_t i = 0; i < meansSize; ++i)
	{
		distance = 0;
		for (my_size_t j = 0; j < dimension; ++j)
		{
			difference = means[i * dimension + j] - data[j * dataSize + id];
			distance += difference * difference;
		}
		if (minDistance > distance)
		{
			minDistance = distance;
			clusterID = i;
		}
	}
	if (id + dataOffset < totalDataSize)
	{
		atomicInc(&counts[clusterID], INT_MAX);
		//assignedClusters[id] = clusterID;
		for (my_size_t j = 0; j < dimension; ++j)
		{
			atomicAdd(&measnSums[clusterID * dimension + j], data[j * dataSize + id]);
		}
	}
}

__global__ void countDivMeansKernel(const uint32_t* counts, value_t* means, const value_t* meansSums, const my_size_t dimension, const uint32_t meansPerBlock)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	means[id] = meansSums[id] / (value_t)counts[blockIdx.x * meansPerBlock + threadIdx.x / dimension];
}