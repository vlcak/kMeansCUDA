#include "fewMeansKernels.cuh"

__global__ void findNearestClusterFewMeansKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension)
{
	unsigned int id = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;  //blockDim.x * (blockIdx.y * gridDim.x + blockIdx.x);
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
		atomicInc(&counts[(threadIdx.x % blockDim.y) * meansSize + clusterID], INT_MAX);
		assignedClusters[id] = clusterID;
		for (size_t j = 0; j < dimension; ++j)
		{
			atomicAdd(&measnSums[(threadIdx.x % blockDim.y) * meansSize * dimension + clusterID * dimension + j], data[id * dimension + j]);
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

// each thread has own copy...delete?
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

__global__ void countDivFewMeansKernelV2(const uint32_t dataSize, const uint32_t meansSize, uint32_t* counts, value_t* means, value_t* meansSums, const uint32_t dimension)
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
	unsigned int id = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;

	extern __shared__ value_t sharedArray[];
	value_t* localSums = (value_t*)&sharedArray[0];
	uint32_t* localCounts = (uint32_t *)&sharedArray[blockDim.y * meansSize * dimension];

	//memory initialization
	// thread x;y will set x+k*blocksizex mean
	for (size_t m = threadIdx.x; m < meansSize; m += blockDim.x)
	{
		for (size_t d = 0; d < dimension; ++d)
		{
			localSums[threadIdx.y * meansSize * dimension + m * dimension + d] = 0;
		}
		localCounts[threadIdx.y * meansSize + m] = 0;
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

	// add data to shared memory
	if (id < dataSize)
	{
		assignedClusters[id] = clusterID;
		for (size_t d = 0; d < dimension; ++d)
		{
			atomicAdd(&localSums[threadIdx.y * dimension * meansSize + clusterID * dimension + d], data[id * dimension + d]);
		}
		atomicInc(&localCounts[threadIdx.y * meansSize + clusterID], INT_MAX);
	}

	__syncthreads();

	for (size_t r = blockDim.y / 2; r > 0; r >>= 1)
	{
		// thread x;y will sum x+k*blocksizex mean
		for (size_t m = threadIdx.x; m < meansSize; m += blockDim.x)
		{
			// thready with y > r will help with reduction - y / r is offset, step is blockdimY / r
			for (size_t d = threadIdx.y / r; d < dimension; d += blockDim.y / r)
			{
				localSums[(threadIdx.y % r) * dimension * meansSize + m * dimension + d] += localSums[((threadIdx.y % r) + r) * dimension * meansSize + m * dimension + d];
			}
			localCounts[(threadIdx.y % r) * meansSize + m] += localCounts[((threadIdx.y % r) + r) * meansSize + m];
		}
		__syncthreads();
	}

	// thread x;y will set x+k*blocksizex mean
	for (size_t m = threadIdx.x; m < meansSize; m += blockDim.x)
	{
		// thready is offset, step is blockdimY
		for (size_t d = threadIdx.y; d < dimension; d += blockDim.y)
		{
			atomicAdd(&measnSums[(blockIdx.x * meansSize + m) * dimension + d], localSums[m * dimension + d]);
		}
		atomicAdd(&counts[blockIdx.x * meansSize + m], localCounts[m]);
	}

}

__global__ void countDivFewMeansKernelV3(const uint32_t dataSize, const uint32_t meansSize, uint32_t* counts, value_t* means, value_t* meansSums, const uint32_t dimension)
{
	//threadID.z - meansID
	//threadID.y - meansCopyID
	//threadID.z - dimension
	int meansID = threadIdx.z + blockDim.z * blockIdx.x;
	for (size_t r = blockDim.y; r > 0; r >>= 1)
	{
		if (threadIdx.y < r)
		{
			meansSums[threadIdx.y * dimension * meansSize + meansID * dimension + threadIdx.x] += meansSums[(threadIdx.y + r) * dimension * meansSize + meansID * dimension + threadIdx.x];
			if (threadIdx.x == 0)
			{
				counts[threadIdx.y * meansSize + meansID] += counts[(threadIdx.y + r) * meansSize + meansID];
			}
		}
		__syncthreads();
	}

	means[meansID * dimension + threadIdx.x] = meansSums[meansID * dimension + threadIdx.x] / (value_t)counts[meansID];
}
