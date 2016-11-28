#include "fewMeansKernels.cuh"

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
		atomicInc(&counts[blockIdx.y * meansSize + clusterID], INT_MAX);
		assignedClusters[id] = clusterID;
		for (size_t j = 0; j < dimension; ++j)
		{
			atomicAdd(&measnSums[blockIdx.y * meansSize * dimension + clusterID * dimension + j], data[id * dimension + j]);
		}
	}
}

__global__ void countDivFewMeansKernel(const uint32_t meansSize, uint32_t* counts, value_t* means, const value_t* meansSums, const uint32_t dimension, const uint32_t cellsCount)
{
	int id = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;

	uint32_t count = 0;

	means[id] = meansSums[id];

	count = counts[blockIdx.x * blockDim.y + threadIdx.y];

	for (size_t i = 1; i < cellsCount; i++)
	{
		means[id] += meansSums[i * dimension * meansSize + id];
		count += counts[i * meansSize + blockIdx.x * blockDim.y + threadIdx.y];
	}

	means[id] /= count;

	if (threadIdx.x == 0)
	{
		counts[blockIdx.x * blockDim.y + threadIdx.y] = count;
	}
}

__global__ void findNearestClusterFewMeansKernelV2(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension)
{
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
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
		++counts[meansSize * id + clusterID];
		assignedClusters[id] = clusterID;
		for (size_t j = 0; j < dimension; ++j)
		{
			measnSums[dimension * (id * meansSize + clusterID) + j] += data[id * dimension + j];
			//atomicAdd(&measnSums[blockIdx.y * meansSize * dimension + clusterID * dimension + j], data[id * dimension + j]);
		}
	}
}

__global__ void countDivFewMeansKernelV2(const uint32_t dataSize, const uint32_t meansSize, uint32_t* counts, value_t* means, value_t* meansSums, const uint32_t dimension, const uint32_t cellsCount)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	for (size_t i = dataSize / 2; i > 0; i >>= 1)
	{
		if (id < i)
		{
			for (size_t j = 0; j < dimension; ++j)
			{
				meansSums[id * dimension * meansSize + j] += meansSums[(id + i) * dimension * meansSize + j];
			}
			counts[id * meansSize] += counts[(id + i) * meansSize];
		}
		__syncthreads();
	}

	means[id] /= counts[id % dimension];
}

__global__ void findNearestClusterFewMeansKernelV3(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension)
{
	unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;

	extern __shared__ value_t sharedArray[];
	value_t* localSums = (value_t*)&sharedArray[0];
	uint8_t* localCounts = (uint8_t *)&sharedArray[blockDim.x * meansSize * dimension];

	//memory initialization
	for (size_t m = 0; m < meansSize; ++m)
	{
		for (size_t d = 0; d < dimension; ++d)
		{
			localSums[threadIdx.x * dimension * meansSize + m * dimension + d] = 0;
		}
		localCounts[threadIdx.x * meansSize + m] = 0;
	}

	value_t minDistance = LLONG_MAX, distance = 0, difference = 0;
	int clusterID = -1;
	for (size_t m = 0; m < meansSize; ++m)
	{
		distance = 0;
		for (size_t d = 0; d < dimension; ++d)
		{
			difference = means[m * dimension + d] - data[id * dimension + d];
			distance += difference * difference;
		}
		if (minDistance > distance)
		{
			minDistance = distance;
			clusterID = m;
		}
	}

	if (id < dataSize)
	{
		assignedClusters[id] = clusterID;
		for (size_t d = 0; d < dimension; ++d)
		{
			localSums[threadIdx.x * dimension * meansSize + clusterID * dimension + d] = data[id * dimension + d];
		}
		localCounts[threadIdx.x * meansSize + clusterID] = 1;
	}

	for (size_t r = blockDim.x / 2; r > 0; r >>= 1)
	{
		if (threadIdx.x < r)
		{
			for (size_t m = 0; m < meansSize; ++m)
			{
				for (size_t d = 0; d < dimension; ++d)
				{
					localSums[threadIdx.x * dimension * meansSize + m * dimension + d] += localSums[(threadIdx.x + r) * dimension * meansSize + m * dimension + d];
				}
				localCounts[threadIdx.x * meansSize + m] += localCounts[(threadIdx.x + r) * meansSize + m];
			}


			//for (size_t j = 0; j < dimension; ++j)
			//{
			//    localSums[threadIdx.x * dimension + j] += localSums[(threadIdx.x + i) * dimension + j];
			//}
			//localCounts[threadIdx.x] += counts[threadIdx.x + i];
		}
		__syncthreads();
	}

	if (threadIdx.x < meansSize)
	{
		for (size_t i = 0; i < dimension; ++i)
		{
			measnSums[(blockIdx.x * meansSize + threadIdx.x) * dimension + i] = localSums[threadIdx.x * dimension + i];
		}
		counts[blockIdx.x * meansSize + threadIdx.x] = localCounts[threadIdx.x];
	}

}

__global__ void countDivFewMeansKernelV3(const uint32_t dataSize, const uint32_t meansSize, uint32_t* counts, value_t* means, value_t* meansSums, const uint32_t dimension, const uint32_t copyCount)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;



	for (size_t r = blockDim.x; r > 0; r >>= 1)
	{
		if (id < r)
		{
			if (threadIdx.x + r < copyCount)
			{
				for (size_t m = 0; m < meansSize; ++m)
				{
					for (size_t d = 0; d < dimension; ++d)
					{
						meansSums[threadIdx.x * dimension * meansSize + m * dimension + d] += meansSums[(threadIdx.x + r) * dimension * meansSize + m * dimension + d];
					}
					counts[threadIdx.x * meansSize + m] += counts[(threadIdx.x + r) * meansSize + m];
				}
			}
		}
		__syncthreads();
	}

	if (id < meansSize * dimension)
	{
		means[id] = meansSums[id] / (value_t)counts[id / dimension];
	}
}
