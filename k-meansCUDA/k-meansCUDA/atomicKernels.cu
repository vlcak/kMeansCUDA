/*
#include "atomicKernels.cuh"

__global__ void findNearestClusterAtomicKernel(const uint32_t meansSize, const value_t *means, value_t *measnSums, const uint32_t dataSize, const value_t* data, uint32_t* counts, uint32_t* assignedClusters, const uint32_t dimension)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
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
    atomicInc(&counts[clusterID], INT32_MAX);
    assignedClusters[id] = clusterID;
    for (size_t j = 0; j < dimension; ++j)
    {
        atomicAdd(&measnSums[clusterID * dimension + j], data[id * dimension + j]);
    }
}

__global__ void countDivMeansKernel(const uint32_t meansSize, const uint32_t* counts, value_t* means, const value_t* meansSums, const uint32_t dimension)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    means[id] = meansSums[id] / (value_t)counts[blockIdx.x];
}


cudaError_t countKMeansAtomic(const uint32_t iterations, const uint32_t dataSize, const value_t* data, const uint32_t meansSize, value_t* means, uint32_t* assignedClusters, uint64_t dimension)
{
    value_t* dev_means = 0, *dev_data = 0, *dev_meansSums = 0, *dev_temp = 0;
    uint32_t* dev_assignedClusters = 0, *dev_counts = 0;
    cudaError_t cudaStatus;

    clock_t start, end;
    start = clock();

    //std::vector<uint32_t> testVector(meansSize);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_means, meansSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_meansSums, meansSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    else
    {
        cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
    }

    cudaStatus = cudaMalloc((void**)&dev_data, dataSize * dimension * sizeof(value_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_assignedClusters, dataSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_counts, meansSize * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    else
    {
        cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_means, means, meansSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_data, data, dataSize * dimension * sizeof(value_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //uint32_t* test = (uint32_t*)calloc(meansSize, sizeof(uint32_t));
    //value_t* testMeans = (value_t*)calloc(meansSize * dimension , sizeof(value_t));

    // Launch a kernel on the GPU with one thread for each element.
    int blockSizeN = 32;
    int nBlocksN = (dataSize - 1) / blockSizeN + 1;
    //int blockSizeM = 16;
    //int nBlocksM = (meansSize - 1) / blockSizeM + 1;
    for (uint32_t i = 0; i < iterations; ++i)
    {
        findNearestClusterAtomicKernel << <nBlocksN, blockSizeN >> >(meansSize, dev_means, dev_meansSums, dataSize, dev_data, dev_counts, dev_assignedClusters, dimension);
        cudaDeviceSynchronize();
        countDivMeansKernel << <meansSize, dimension >> >(meansSize, dev_counts, dev_means, dev_meansSums, dimension);
        cudaDeviceSynchronize();

        cudaMemset(dev_meansSums, 0, meansSize * dimension * sizeof(value_t));
        cudaMemset(dev_counts, 0, meansSize * sizeof(uint32_t));
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(means, dev_means, meansSize * dimension * sizeof(value_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(assignedClusters, dev_assignedClusters, dataSize * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



Error:
    cudaFree(dev_data);
    cudaFree(dev_means);
    cudaFree(dev_meansSums);
    cudaFree(dev_assignedClusters);
    cudaFree(dev_counts);


    end = clock();
    std::cout << "Time required for execution: "
        << (double)(end - start) / CLOCKS_PER_SEC
        << " seconds." << "\n\n";

    return cudaStatus;
}*/
