#include "warpPerPointKernel.cuh"

template <typename T>
__device__  T min(T a, T b, int offset)
{
    return a < b ? a : b;
}


__global__ void findNearestWarpPerPointKernel(const value_t* __restrict__ means, value_t *meansSums, const value_t* __restrict__ data, uint32_t* counts, const my_size_t dimension)
{
    extern __shared__ value_t sharedArray[];
    value_t* distances = (value_t*)&sharedArray[blockDim.x * threadIdx.y]; // array of distances to all means
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
    //    minDistance = distance;
    //    minMeanID = m;
    //}

    //}

    // copy distance to mean to shared memory
    distances[threadIdx.x] = distance;
    //minMeansIds[threadIdx.x] = minMeanID;
    // find the nearest mean id
    for (my_size_t t = 0; t < blockDim.x; ++t)
    {
        // >= guarantee that all threads will compute with same mean (means with higher id are preffered)
        if (distance >= distances[t])
        {
            distance = distances[t];
            minMeanId = t;
        }
    }

    // store values to global memory
    if (threadIdx.x == 0)
    {
        atomicInc(&counts[minMeanId], INT32_MAX);
    }

    // utilize all threads
    for (my_size_t d = threadIdx.x; d < dimension; d += blockDim.x)
    {
        atomicAdd(&meansSums[minMeanId * dimension + d], data[pointID * dimension + d]);
    }
}

__global__ void findNearestWarpPerPointSMKernel(const value_t* __restrict__ means, value_t* __restrict__ meansSums, const value_t* __restrict__ data, uint32_t* counts, const my_size_t dimension)
{
    extern __shared__ value_t sharedArray[];
    value_t* point = (value_t*)&sharedArray[threadIdx.y * dimension];
    value_t* distances = (value_t*)&sharedArray[blockDim.y * dimension + blockDim.x * threadIdx.y];
    
    unsigned int pointID = threadIdx.y + blockIdx.x * blockDim.x * blockDim.y
        , minMeanId = threadIdx.x;

    // point is copied to shared memory - coalesced acces to global memory, bank-safe save to shared
    for (my_size_t d = threadIdx.x; d < dimension; d += blockDim.x)
    {
        point[d] = data[pointID * dimension + d];
    }

    value_t distance = 0
        , difference = 0;

    for (my_size_t d = 0; d < dimension; ++d)
    {
		// all threads read the same value - multicast
        difference = means[threadIdx.x * dimension + d] - point[d];
        distance += difference * difference;
    }
    distances[threadIdx.x] = distance;

    __syncthreads();
    for (my_size_t t = 0; t < blockDim.x; ++t)
    {
        // >= guarantee that all threads will compute with same mean (mean with higher id is preffered)
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

    // utilize all threads
    for (my_size_t d = threadIdx.x; d < dimension; d += blockDim.x)
    {
        atomicAdd(&meansSums[minMeanId * dimension + d], data[pointID * dimension + d]);
    }
}

#if __CUDA_ARCH__ >= 300
__global__ void findNearestWarpPerPointShuffleKernel(const value_t* __restrict__ means, value_t *meansSums, const value_t* __restrict__ data, uint32_t* counts, const my_size_t dimension)
{
    int pointID = threadIdx.y + blockIdx.x * blockDim.x * blockDim.y;
    unsigned int minMeanId = threadIdx.x
          , tempMean = 0;
    value_t distance = 0
             , difference = 0
          , tempDistance = 0;
    //for (my_size_t m = threadIdx.x; m < meansSize; m += blockDim.x)
    //{
    for (my_size_t d = 0; d < dimension; ++d)
    {
        difference = means[threadIdx.x * dimension + d] - data[pointID * dimension + d];
        distance += difference * difference;
    }
    //}

    //int nearestMeanID = threadIdx.x;

    for (size_t i = 1; i < warpSize / 2; i <<= 1)
    {
        tempDistance = min(distance, __shfl_xor(distance, i));
        tempMean = __shfl_xor(minMeanId, i);
        if (tempDistance < distance)
        {
            distance = tempDistance;
            minMeanId = tempMean;
        }
        else if (tempDistance == distance)
        {
            minMeanId = min(tempMean, minMeanId);
        }
    }
    //distance = min(distance, __shfl_xor(distance, 2));
    //distance = min(distance, __shfl_xor(distance, 4));
    //distance = min(distance, __shfl_xor(distance, 8));
    //distance = min(distance, __shfl_xor(distance, 16));
    //if (warpSize > 32) distance = min(distance, __shfl_xor(distance, 32));

    if (threadIdx.x == 0)
    {
        atomicInc(&counts[minMeanId], INT_MAX);
    }

    // utilize all threads
    for (my_size_t d = threadIdx.x; d < dimension; d += blockDim.x)
    {
        atomicAdd(&meansSums[minMeanId * dimension + d], data[pointID * dimension + d]);
    }

}
#endif
