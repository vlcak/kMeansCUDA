#include "simpleKernels.cuh"

__global__ void findNearestClusterKernel(const my_size_t meansSize, const value_t* __restrict__ means, const value_t* __restrict__ data, uint32_t* __restrict__ assignedClusters, const my_size_t dimension)
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

__global__ void countNewMeansKernel(const uint32_t* __restrict__ assignedClusters, const my_size_t dataSize, const value_t* __restrict__ data, value_t* __restrict__ means, const my_size_t dimension)
{
    int id = threadIdx.y + blockIdx.x * blockDim.y;
    int idOffset = id * dimension + threadIdx.x;
    uint32_t count = 0;
    // set mean computed by threadId.y
    means[idOffset] = 0;
    for (my_size_t i = 0; i < dataSize; ++i)
    {
        if (assignedClusters[i] == id)
        {
            means[idOffset] += data[i * dimension + threadIdx.x];
            ++count;
        }
    }
    means[idOffset] /= count;
}