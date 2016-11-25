#include "manyDimensionsKernels.cuh"

__global__ void findNearestClusterManyDimKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension)
{
	//int id = threadIdx.x;
	value_t minDistance = LLONG_MAX, difference = 0, distance = 0;
	int clusterID = -1;
	extern __shared__ value_t distances[];

	for (size_t i = 0; i < meansSize; ++i)
	{
		for (size_t j = threadIdx.x; j < dimension; j += blockDim.x)
		{
			difference = means[i * dimension + j] - data[blockIdx.x * dimension + j];
			distance += difference * difference;
		}
		distances[threadIdx.x] = distance;

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

		if ((minDistance > distances[0]))
		{
			minDistance = distances[0];
			clusterID = i;
		}
	}

	if (threadIdx.x == 0)
	{
		atomicInc(&counts[clusterID], INT_MAX);
		assignedClusters[blockIdx.x] = clusterID;
	}
	for (size_t j = threadIdx.x; j < dimension; j += blockDim.x)
	{
		atomicAdd(&measnSums[clusterID * dimension + j], data[blockIdx.x * dimension + j]);
	}
}

__global__ void findNearestClusterManyDimUnrolledKernel(const uint32_t meansSize, const value_t *means, value_t *meansSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension, const uint16_t blockSize)
{
	//int id = threadIdx.x;
	value_t minDistance = LLONG_MAX, difference = 0, distance = 0;
	int clusterID = -1;
	extern __shared__ value_t distances[];

	for (size_t i = 0; i < meansSize; ++i)
	{
		for (size_t j = threadIdx.x; j < dimension; j += blockDim.x)
		{
			difference = means[i * dimension + j] - data[blockIdx.x * dimension + j];
			distance += difference * difference;
		}
		distances[threadIdx.x] = distance;

		//sum distances in block
		__syncthreads();


		//if (blockSize >= 512) { if (threadIdx.x < 256) { distances[threadIdx.x] += distances[threadIdx.x + 256]; } __syncthreads(); }
		//if (blockSize >= 256) { if (threadIdx.x < 128) { distances[threadIdx.x] += distances[threadIdx.x + 128]; } __syncthreads(); }
		//if (blockSize >= 128) { if (threadIdx.x <  64) { distances[threadIdx.x] += distances[threadIdx.x +  64]; } __syncthreads(); }

		if (threadIdx.x < blockDim.x / 2)
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
		atomicInc(&counts[clusterID], INT_MAX);
		assignedClusters[blockIdx.x] = clusterID;
	}

	for (size_t j = threadIdx.x; j < dimension; j += blockDim.x)
	{
		atomicAdd(&meansSums[clusterID * dimension + j], data[blockIdx.x * dimension + j]);
	}
}

__global__ void findNearestClusterManyDimShuffleKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension)
{
	//int id = threadIdx.x;
	value_t minDistance = LLONG_MAX, difference = 0, distance = 0;
	int clusterID = -1;
	extern __shared__ value_t distances[];

	for (size_t i = 0; i < meansSize; ++i)
	{
		for (size_t j = threadIdx.x; j < dimension; j += blockDim.x)
		{
			difference = means[i * blockDim.x + j] - data[blockIdx.x * blockDim.x + j];
			distance += difference * difference;
		}
		distances[threadIdx.x] = distance;

		//sum distances in block
		__syncthreads();


		//if (blockSize >= 512) { if (threadIdx.x < 256) { distances[threadIdx.x] += distances[threadIdx.x + 256]; } __syncthreads(); }
		//if (blockSize >= 256) { if (threadIdx.x < 128) { distances[threadIdx.x] += distances[threadIdx.x + 128]; } __syncthreads(); }
		//if (blockSize >= 128) { if (threadIdx.x <  64) { distances[threadIdx.x] += distances[threadIdx.x +  64]; } __syncthreads(); }

		//for (int offset = warpSize / 2; offset > 0; offset /= 2)
		//if (warpSize > 32) distance += __shfl_down(distance, 32);
		//distance += __shfl_down(distance, 16);
		//distance += __shfl_down(distance, 8);
		//distance += __shfl_down(distance, 4);
		//distance += __shfl_down(distance, 2);
		//distance += __shfl_down(distance, 1);

		if ((minDistance > distance))
		{
			minDistance = distances[0];
			clusterID = i;
		}
	}

	if (threadIdx.x == 0)
	{
		atomicInc(&counts[clusterID], INT_MAX);
		assignedClusters[blockIdx.x] = clusterID;
	}

	for (size_t j = threadIdx.x; j < dimension; j += blockDim.x)
	{
		atomicAdd(&measnSums[clusterID * blockDim.x + j], data[blockIdx.x * blockDim.x + j]);
	}
}