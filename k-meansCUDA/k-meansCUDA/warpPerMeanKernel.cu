#include "warpPerMeanKernel.cuh"

__global__ void findNearestClusterWarpPerMeanThreadPerPointKernel(const value_t* __restrict__ means, const my_size_t dataSize, const value_t* __restrict__ data, uint32_t* locks, value_t* distances, uint32_t* assignedClusters, const my_size_t dimension)
{
    value_t *localMean = new value_t[dimension];
    value_t distance = 0, difference = 0;
    size_t pointIndex = 0
         , meanID = blockIdx.x * blockDim.y + threadIdx.y
         , offsetPerMean;
    // copy mean to local memory
    for (my_size_t i = threadIdx.x; i < dimension; i += blockDim.x)
    {
        localMean[i] = means[meanID * dimension + i];
    }
    // offset to distribute access to points - each mean starting from meanID-th part of points
    offsetPerMean = meanID * (dataSize / (gridDim.x * blockDim.y));
    // iterate through all points
    for (my_size_t i = threadIdx.x; i < dataSize; i += blockDim.x)
    {
        pointIndex = (i + offsetPerMean) % dataSize;
        distance = 0;
        for (my_size_t j = 0; j < dimension; ++j)
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

#if __CUDA_ARCH__ >= 300
__global__ void findNearestClusterWarpPerMeanThreadPerDimensionKernel(const value_t* __restrict__ means, const my_size_t dataSize, const value_t* __restrict__ data, uint32_t* locks, value_t* distances, uint32_t* assignedClusters, const my_size_t dimension)
{
    value_t *localMean = new value_t[dimension];
    value_t distance = 0, difference = 0;
    size_t pointIndex = 0, meanID, offsetPerMean;
    meanID = blockIdx.x * blockDim.y + threadIdx.y;
    // copy mean to local memory
    for (my_size_t i = threadIdx.x; i < dimension; i += blockDim.x)
    {
        localMean[i] = means[meanID * dimension + i];
    }
    // offset to distribute access to points - each mean starting from meanID-th part of points
    offsetPerMean = meanID * (dataSize / (gridDim.x * blockDim.y));
    // iterate through all points
    for (my_size_t i = 0; i < dataSize; ++i)
    {
        pointIndex = (i + offsetPerMean) % dataSize;
        distance = 0;
        // each thread computes dimension
        difference = localMean[threadIdx.x] - data[pointIndex * dimension + threadIdx.x];
        distance += difference * difference;

        if (warpSize > 32) distance += __shfl_down(distance, 32);
        distance += __shfl_down(distance, 16);
        distance += __shfl_down(distance, 8);
        distance += __shfl_down(distance, 4);
        distance += __shfl_down(distance, 2);
        distance += __shfl_down(distance, 1);

        if (threadIdx.x == 0 && distance < distances[pointIndex])
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
#endif

__global__ void countNewMeansWarpPerMeansKernel(value_t* __restrict__ newMeans, const my_size_t dataSize, const value_t* __restrict__ data, const uint32_t* __restrict__ assignedClusters, const my_size_t dimension)
{
    uint32_t meanID = blockIdx.x * blockDim.y + threadIdx.y
           , assignedPoints = 0;
    value_t coordinateSum = 0;
    for (my_size_t i = 0; i < dataSize; ++i)
    {
        if (meanID == assignedClusters[i])
        {
            ++assignedPoints;
            coordinateSum += data[i * dimension + threadIdx.x];
        }
    }

    newMeans[meanID * dimension + threadIdx.x] = coordinateSum / assignedPoints;
}