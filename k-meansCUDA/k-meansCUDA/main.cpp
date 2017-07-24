#include "atomicTasks.cuh"
#include "fewMeansTasks.cuh"
#include "manyDimensionsTasks.cuh"
#include "warpPerMeanTasks.cuh"
#include "warpPerPointTasks.cuh"
#include "simpleTasks.cuh"
#include "helpers.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <stdio.h>

const char* ATOMIC_ALG = "ATOMIC";
const char* ATOMIC_BIG_ALG = "ATOMIC_B";
const char* ATOMIC_TRANSPOSED_ALG = "ATOMIC_T";
const char* FEW_MEANS_ALG = "FEW_MEANS";
const char* FEW_MEANS_V2_ALG = "FEW_MEANS_V2";
const char* FEW_MEANS_V3_ALG = "FEW_MEANS_V3";
const char* MANY_DIMS_ALG = "MANY_DIMS";
const char* SIMPLE_ALG = "SIMPLE";
const char* WARP_PER_MEAN_ALG = "WARP_PER_MEAN";
const char* WARP_PER_POINT_ALG = "WARP_PER_POINT";

const int INVALID_ALGORITHM = 20000;

void usage()
{
    std::cout << 
"Usage:" << std::endl << "\
    kmeans <data_file> <k> <iterations> ALGORITHM [device ID] [Algorithm specific parameters]" << std::endl << "\
Possible ALGORITHM values:" << std::endl << "\
    " << ATOMIC_ALG << " [--transposed|--shared] normal otherwise" << std::endl << "\
    " << ATOMIC_BIG_ALG << std::endl << "\
    " << FEW_MEANS_ALG << " [SHARED MEMORY DUPLICITY]" << std::endl << "\
    " << FEW_MEANS_V2_ALG << std::endl << "\
    " << FEW_MEANS_V3_ALG << " [SHARED MEMORY DUPLICTY] [GLOBAL MEMORY DUPLICITY]" << std::endl << "\
    " << MANY_DIMS_ALG << " [--unrolled|--shuffle] normal otherwise" << std::endl << "\
    " << SIMPLE_ALG << std::endl << "\
    " << WARP_PER_MEAN_ALG << " [--dimension] normal otherwise" << std::endl << "\
    " << WARP_PER_POINT_ALG << " [--sharedMemory|--shuffle] normal otherwise" << std::endl << "\
    " << std::endl;
}

int main(int argc, const char* argv[])
{
    if (argc >= 5 && argc <= 8)
    {
        std::string file_name(argv[1]);
		std::string outputs_fileNameSuffix = file_name;
        //std::string means_file_name = file_name;
        //std::string clusters_file_name = file_name;
        // Add vectors in parallel.
		std::string algorithm(argv[4]);
        int dataPos = file_name.find_last_of("/") + 1;
		outputs_fileNameSuffix.erase(0, dataPos + std::string("data").length());
		outputs_fileNameSuffix.insert(0, "_");
		if (argc > 7) outputs_fileNameSuffix.insert(0, argv[7]);
		if (argc > 6) outputs_fileNameSuffix.insert(0, argv[6]);
		outputs_fileNameSuffix.insert(0, algorithm);
		std::string means_file_name = outputs_fileNameSuffix;
		std::string clusters_file_name = outputs_fileNameSuffix;
        means_file_name.insert(0, "means_");
		clusters_file_name.insert(0, "cluster_");
        std::string s_k(argv[2]);
        std::string s_iterations(argv[3]);
        uint32_t k = lexical_cast<uint32_t>(s_k);
        uint32_t iterations = lexical_cast<uint32_t>(s_iterations);
        int deviceID = argc > 5 ? lexical_cast<int>(argv[5]) : 0;
        verifyDeviceID(deviceID);
        std::cout << "Computing on CUDA device #" << deviceID << std::endl;
        DEVICE_ID = deviceID;
        printDeviceInfo(DEVICE_ID);

        uint64_t dataSize;
        uint64_t dimension;
        std::unique_ptr<value_t[]> data = loadData(file_name, dataSize, dimension);
        std::unique_ptr<value_t[]> means = std::make_unique<value_t[]>(k * dimension);
        std::unique_ptr<uint32_t[]> assignedClusters = std::make_unique<uint32_t[]>(dataSize * dimension);
        std::memcpy(means.get(), data.get(), k * dimension * sizeof(value_t));

        int cudaStatus = INVALID_ALGORITHM;
        if (algorithm == ATOMIC_ALG)
			cudaStatus = countKMeansAtomic(iterations, dataSize, data.get(), k, means.get(), assignedClusters.get(), dimension, argc > 6 ? argv[6] : "");
        if (algorithm == ATOMIC_BIG_ALG)
            cudaStatus = countKMeansBIGDataAtomic(iterations, dataSize, data.get(), k, means.get(), assignedClusters.get(), dimension);
        if (algorithm == FEW_MEANS_ALG)
            cudaStatus = countKMeansFewMeans(iterations, dataSize, data.get(), k, means.get(), assignedClusters.get(), dimension, lexical_cast<uint32_t>(argv[6]));
        if (algorithm == FEW_MEANS_V2_ALG)
            cudaStatus = countKMeansFewMeansV2(iterations, dataSize, data.get(), k, means.get(), assignedClusters.get(), dimension);
        if (algorithm == FEW_MEANS_V3_ALG)
            cudaStatus = countKMeansFewMeansV3(iterations, dataSize, data.get(), k, means.get(), assignedClusters.get(), dimension, lexical_cast<uint32_t>(argv[6]), lexical_cast<uint32_t>(argv[7]));
        if (algorithm == MANY_DIMS_ALG)
            cudaStatus = countKMeansManyDims(iterations, dataSize, data.get(), k, means.get(), assignedClusters.get(), dimension, argc > 6 ? argv[6] : "");
        if (algorithm == SIMPLE_ALG)
            cudaStatus = countKMeansSimple(iterations, dataSize, data.get(), k, means.get(), assignedClusters.get(), dimension);
        if (algorithm == WARP_PER_MEAN_ALG)
            cudaStatus = countKMeansWarpPerMean(iterations, dataSize, data.get(), k, means.get(), assignedClusters.get(), dimension, argc > 6 ? argv[6] : "");
        if (algorithm == WARP_PER_POINT_ALG)
            cudaStatus = countKMeansWarpPerPoint(iterations, dataSize, data.get(), k, means.get(), assignedClusters.get(), dimension, argc > 6 ? argv[6] : "");

        switch (cudaStatus)
        {
        case INVALID_ALGORITHM:
            std::cerr << "invalid algorithm!" << std::endl;
            usage();
            return 1;
        case cudaSuccess:
            std::cout << "CUDA computation successful!" << std::endl;
            break;
        default:
            std::cerr << "CUDA error: " << cudaStatus << std::endl;
            return 1;
        }

        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            std::cerr << "cudaDeviceReset failed!" << std::endl;
            return 1;
        }

        saveResults(means_file_name, clusters_file_name, k, means, dataSize, data, assignedClusters, dimension);

        return 0;
    }
    usage();
    return 1;
}

#pragma endregion