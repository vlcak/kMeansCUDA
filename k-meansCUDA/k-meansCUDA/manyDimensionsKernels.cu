#include "manyDimensionsKernels.cuh"

__global__ void findNearestClusterManyDimKernel(const my_size_t meansSize, const value_t* __restrict__ means, value_t *meansSums, const value_t* __restrict__ data, uint32_t* counts, uint32_t* __restrict__ assignedClusters, const my_size_t dimension)
{
    //int id = threadIdx.x;
    value_t minDistance = LLONG_MAX, difference = 0, distance = 0;
    int clusterID = -1;
    extern __shared__ value_t distances[];

    for (my_size_t i = 0; i < meansSize; ++i)
    {
        for (my_size_t j = threadIdx.x; j < dimension; j += blockDim.x)
        {
            // means are addressed for each point equally
            // data are addressed following (offset of points counted by previous blocks + offset of points counted by previous warps)
            difference = means[i * dimension + j] - data[blockIdx.x * blockDim.y * dimension + threadIdx.y * dimension + j];
            distance += difference * difference;
        }
        // offset for different warps + offset for current thread
        distances[threadIdx.y * blockDim.x + threadIdx.x] = distance;

        //sum distances in warp
        // warp is using only half of threads, but it cant count distances for next warp because of synchronization (or sync)
        // __syncthreads();
        for (my_size_t j = blockDim.x / 2; j > 0; j >>= 1)
        {
            if (threadIdx.x < j)
            {
                distances[threadIdx.y * blockDim.x + threadIdx.x] += distances[threadIdx.y * blockDim.x + threadIdx.x + j];
            }
            // this is not needed if we dont sync threads
            // __syncthreads();
        }

        if ((minDistance > distances[threadIdx.y * blockDim.x]))
        {
            minDistance = distances[threadIdx.y * blockDim.x];
            clusterID = i;
        }
    }

    if (threadIdx.x == 0)
    {
        atomicInc(&counts[clusterID], INT_MAX);
        // number of points counted in different blocks + number of points counted in different warps
        assignedClusters[blockIdx.x * blockDim.y + threadIdx.y] = clusterID;
    }
    for (my_size_t j = threadIdx.x; j < dimension; j += blockDim.x)
    {
        atomicAdd(&meansSums[clusterID * dimension + j], data[blockIdx.x * blockDim.y * dimension + threadIdx.y * dimension + j]);
    }
}

__global__ void findNearestClusterManyDimUnrolledKernel(const my_size_t meansSize, const value_t* __restrict__ means, value_t *meansSums, const value_t* __restrict__ data, uint32_t* counts, uint32_t* __restrict__ assignedClusters, const my_size_t dimension)
{
    //int id = threadIdx.x;
    value_t minDistance = LLONG_MAX, difference = 0, distance = 0;
    int clusterID = -1;
    extern __shared__ value_t distances[];

    for (my_size_t i = 0; i < meansSize; ++i)
    {
        for (my_size_t j = threadIdx.x; j < dimension; j += blockDim.x)
        {
            // means are addressed for each point equally
            // data are addressed following (offset of points counted by previous blocks + offset of points counted by previous warps)
            difference = means[i * dimension + j] - data[blockIdx.x * blockDim.y * dimension + threadIdx.y * dimension + j];
            distance += difference * difference;
        }
        // offset for different warps + offset for current thread
        distances[threadIdx.y * blockDim.x + threadIdx.x] = distance;

        //sum distances in block
        //if (blockSize >= 512) { if (threadIdx.x < 256) { distances[threadIdx.x] += distances[threadIdx.x + 256]; } __syncthreads(); }
        //if (blockSize >= 256) { if (threadIdx.x < 128) { distances[threadIdx.x] += distances[threadIdx.x + 128]; } __syncthreads(); }
        //if (blockSize >= 128) { if (threadIdx.x <  64) { distances[threadIdx.x] += distances[threadIdx.x +  64]; } __syncthreads(); }

        //if (threadIdx.x < blockDim.x / 2)
        //{
        int threadDistanceID = threadIdx.y * blockDim.x + threadIdx.x;
        if (blockDim.x >= 64) distances[threadDistanceID] += distances[threadDistanceID + 32];
        if (blockDim.x >= 32) distances[threadDistanceID] += distances[threadDistanceID + 16];
        if (blockDim.x >= 16) distances[threadDistanceID] += distances[threadDistanceID + 8];
        if (blockDim.x >= 8) distances[threadDistanceID] += distances[threadDistanceID + 4];
        if (blockDim.x >= 4) distances[threadDistanceID] += distances[threadDistanceID + 2];
        if (blockDim.x >= 2) distances[threadDistanceID] += distances[threadDistanceID + 1];
        //}


        if ((minDistance > distances[threadIdx.y * blockDim.x]))
        {
            minDistance = distances[threadIdx.y * blockDim.x];
            clusterID = i;
        }
    }

    if (threadIdx.x == 0)
    {
        atomicInc(&counts[clusterID], INT_MAX);
        // number of points counted in different blocks + number of points counted in different warps
        assignedClusters[blockIdx.x * blockDim.y + threadIdx.y] = clusterID;
    }
    for (my_size_t j = threadIdx.x; j < dimension; j += blockDim.x)
    {
        atomicAdd(&meansSums[clusterID * dimension + j], data[blockIdx.x * blockDim.y * dimension + threadIdx.y * dimension + j]);
    }
}

#if __CUDA_ARCH__ >= 300
__global__ void findNearestClusterManyDimShuffleKernel(const my_size_t meansSize, const value_t* __restrict__ means, value_t *meansSums, const value_t* __restrict__ data, uint32_t* counts, uint32_t* __restrict__ assignedClusters, const my_size_t dimension)
{
    //int id = threadIdx.x;
    value_t minDistance = LLONG_MAX, difference = 0, distance = 0;
    int clusterID = -1;

    for (my_size_t i = 0; i < meansSize; ++i)
    {
        for (my_size_t j = threadIdx.x; j < dimension; j += blockDim.x)
        {
            // means are addressed for each point equally
            // data are addressed following (offset of points counted by previous blocks + offset of points counted by previous warps)
            difference = means[i * dimension + j] - data[blockIdx.x * blockDim.y * dimension + threadIdx.y * dimension + j];
            distance += difference * difference;
        }

        //sum distances in block
        //if (blockSize >= 512) { if (threadIdx.x < 256) { distances[threadIdx.x] += distances[threadIdx.x + 256]; } __syncthreads(); }
        //if (blockSize >= 256) { if (threadIdx.x < 128) { distances[threadIdx.x] += distances[threadIdx.x + 128]; } __syncthreads(); }
        //if (blockSize >= 128) { if (threadIdx.x <  64) { distances[threadIdx.x] += distances[threadIdx.x +  64]; } __syncthreads(); }

        //for (int offset = warpSize / 2; offset > 0; offset /= 2)

        distance += __shfl_xor(distance, 1);
        distance += __shfl_xor(distance, 2);
        distance += __shfl_xor(distance, 4);
        distance += __shfl_xor(distance, 8);
        distance += __shfl_xor(distance, 16);
        if (warpSize > 32) distance += __shfl_xor(distance, 32);

        /*if (warpSize > 32) distance += __shfl_down(distance, 32);
        distance += __shfl_down(distance, 16);
        distance += __shfl_down(distance, 8);
        distance += __shfl_down(distance, 4);
        distance += __shfl_down(distance, 2);
        distance += __shfl_down(distance, 1);*/

        if ((minDistance > distance))
        {
            minDistance = distance;
            clusterID = i;
        }
    }

    if (threadIdx.x == 0)
    {
        atomicInc(&counts[clusterID], INT_MAX);
        // number of points counted in different blocks + number of points counted in different warps
        assignedClusters[blockIdx.x * blockDim.y + threadIdx.y] = clusterID;
    }
    for (my_size_t j = threadIdx.x; j < dimension; j += blockDim.x)
    {
        atomicAdd(&meansSums[clusterID * dimension + j], data[blockIdx.x * blockDim.y * dimension + threadIdx.y * dimension + j]);
    }
}
#endif