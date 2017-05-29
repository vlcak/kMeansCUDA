#include "simpleKernels.cuh"

__global__ void findNearestClusterKernel(const my_size_t meansSize, const value_t *means, const value_t* data, uint32_t* assignedClusters, const my_size_t dimension)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	value_t minDistance = LLONG_MAX, distance = 0, difference = 0;
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
			assignedClusters[id] = i;
		}
	}
}

__global__ void countNewMeansKernel(uint32_t* assignedClusters, const my_size_t dataSize, const value_t* data, value_t* means, const my_size_t dimension, uint32_t* test)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int idOffset = id * dimension;
	uint32_t count = 0;
	for (my_size_t i = idOffset; i < idOffset + dimension; ++i)
	{
		means[i] = 0;
	}
	for (my_size_t i = 0; i < dataSize; ++i)
	{
		if (assignedClusters[i] == id)
		{
			for (my_size_t j = 0; j < dimension; ++j)
			{
				means[idOffset + j] += data[i * dimension + j];
			}
			++count;
		}
	}
	for (my_size_t i = idOffset; i < idOffset + dimension; ++i)
	{
		means[i] /= count;
	}
	test[id] = count;
}