#include "warpPerPointKernel.cuh"

template <typename T>
__device__  T min(T a, T b, int offset)
{
	return a < b ? a : b;
}


__global__ void findNearestWarpPerPointKernel(const my_size_t meansSize, const value_t *means, value_t *meansSums, const value_t* data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize)
{
	extern __shared__ value_t sharedArray[];
	value_t* distances = (value_t*)&sharedArray[blockDim.x * threadIdx.y + threadIdx.x];
	//int sizeCoef = sizeof(unsigned int) / sizeof(value_t);
	//unsigned int* minMeansIds = (unsigned int*)&sharedArray[blockDim.y * dimension + blockDim.x * blockDim.y + blockDim.x * threadIdx.y * sizeCoef + threadIdx.x * sizeCoef];

	unsigned int pointID = threadIdx.y + blockIdx.x * blockDim.x * blockDim.y
		, minMeanId = threadIdx.x;
	value_t distance = 0
		, difference = 0;
	//for (my_size_t m = threadIdx.x; m < meansSize; m += blockDim.x)
	//{
	for (my_size_t d = 0; d < dimension; ++d)
	{
		difference = means[threadIdx.x * dimension + d] - data[pointID * dimension + d];
		distance += difference * difference;
	}
	//if (minDistance > distance)
	//{
	//	minDistance = distance;
	//	minMeanID = m;
	//}

	//}
	distances[threadIdx.x] = distance;
	//minMeansIds[threadIdx.x] = minMeanID;
	for (my_size_t t = 0; t < blockDim.x; ++t)
	{
		// <= guarantee that all threads will compute with same mean
		if (distance >= distances[t])
		{
			distance = distances[t];
			minMeanId = t;
		}
	}

	if (threadIdx.x == 0)
	{
		atomicInc(&counts[minMeanId], INT_MAX);
	}

	for (my_size_t d = threadIdx.x; d < dimension; d += blockDim.x)
	{
		atomicAdd(&meansSums[minMeanId * dimension + d], data[pointID * dimension + d]);
	}
}

__global__ void findNearestWarpPerPointSMKernel(const my_size_t meansSize, const value_t *means, value_t *meansSums, const value_t* data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize)
{
	extern __shared__ value_t sharedArray[];
	value_t* point = (value_t*)&sharedArray[threadIdx.y * dimension];
	value_t* distances = (value_t*)&sharedArray[blockDim.y * dimension + blockDim.x * threadIdx.y + threadIdx.x];
	
	unsigned int pointID = threadIdx.y + blockIdx.x * blockDim.x * blockDim.y
		, minMeanId = threadIdx.x;
	for (my_size_t d = 0; d < dimension; d += blockDim.x)
	{
		point[d] = data[pointID * dimension + d];
	}
	value_t distance = 0
		, difference = 0;

	for (my_size_t d = 0; d < dimension; ++d)
	{
		difference = means[threadIdx.x * dimension + d] - point[d];
		distance += difference * difference;
	}
	distances[threadIdx.x] = distance;

	__syncthreads();
	for (my_size_t t = 0; t < blockDim.x; ++t)
	{
		// <= guarantee that all threads will compute with same mean
		if (distance >= distances[t])
		{
			distance = distances[t];
			minMeanId = t;
		}
	}

	if (threadIdx.x == 0)
	{
		atomicInc(&counts[minMeanId], INT_MAX);
	}

	for (my_size_t d = threadIdx.x; d < dimension; d += blockDim.x)
	{
		atomicAdd(&meansSums[minMeanId * dimension + d], data[pointID * dimension + d]);
	}
}

__global__ void findNearestWarpPerPointKernelShuffle(const my_size_t meansSize, const value_t *means, value_t *measnSums, const value_t* data, uint32_t* counts, const my_size_t dimension, const uint32_t dataOffset, const uint32_t totalDataSize)
{
	int threadID = threadIdx.y + blockIdx.x * blockDim.x * blockDim.y;
	value_t distance = 0, difference = 0;
	for (my_size_t m = threadIdx.x; m < meansSize; m += blockDim.x)
	{
		for (my_size_t d = 0; d < dimension; ++d)
		{
			difference = means[m * dimension + d] - data[threadID * dimension + d];
			distance += difference * difference;
		}
	}

	int nearestMeanID = threadIdx.x;

	distance = min(distance, __shfl_xor(distance, 1));
	distance = min(distance, __shfl_xor(distance, 2));
	distance = min(distance, __shfl_xor(distance, 4));
	distance = min(distance, __shfl_xor(distance, 8));
	distance = min(distance, __shfl_xor(distance, 16));
	if (warpSize > 32) distance = min(distance, __shfl_xor(distance, 32));

	// TODO what next?

}
