#include "warpPerMeanKernel.cuh"

__global__ void findNearestClusterWarpPerMeanKernel(const value_t *means, const uint32_t dataSize, const value_t* data, uint32_t* locks, value_t* distances, uint32_t* assignedClusters, const uint32_t dimension)
{
	value_t *localMean = new value_t[dimension];
	value_t distance = 0, difference = 0;
	size_t pointIndex = 0, offsetPerMean;
	// copy mean to local memory
	for (size_t i = threadIdx.x; i < dimension; i += blockDim.x)
	{
		localMean[i] = means[blockIdx.x * dimension + i];
	}
	offsetPerMean = blockIdx.x * (dataSize / gridDim.x);
	// iterate through all points
	for (size_t i = threadIdx.x; i < dataSize; i += blockDim.x)
	{
		pointIndex = (i + offsetPerMean) % dataSize;
		distance = 0;
		for (size_t j = 0; j < dimension; ++j)
		{
			difference = localMean[j] - data[pointIndex * dimension + j];
			distance += difference * difference;
		}
		if (difference < distances[pointIndex])
		{
			// try lock
			while (locks[pointIndex] != blockIdx.x)
			{
				atomicCAS(&locks[pointIndex], UINT_MAX, blockIdx.x);
			}
			// double check pattern
			if (difference < distances[pointIndex])
			{
				distances[pointIndex] = distance;
				assignedClusters[pointIndex] = blockIdx.x;
			}
			// unlock
			locks[pointIndex] = UINT_MAX;
		}
	}

	delete localMean;
}

__global__ void countNewMeansWarpPerMeansKernel(value_t* newMeans, const uint32_t dataSize, const value_t* data, const uint32_t* assignedClusters, const uint32_t dimension)
{
	uint32_t meanID = blockIdx.x * blockDim.y + blockIdx.y
		   , assignedPoints = 0;
	value_t coordinateSum = 0;
	for (size_t i = 0; i < dataSize; ++i)
	{
		if (meanID == assignedClusters[i])
		{
			++assignedPoints;
			coordinateSum += data[i * dimension + threadIdx.x];
		}
	}

	newMeans[meanID * dimension + threadIdx.x] = coordinateSum / assignedPoints;
}