#include "atomicTasks.cuh"
#include "fewMeansTasks.cuh"
#include "manyDimensionsTasks.cuh"
#include "warpPerMeanTasks.cuh"
#include "warpPerPointTasks.cuh"
#include "simpleTasks.cuh"
#include "helpers.h"

#include <iostream>

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

void usage()
{
    std::cout << 
"Usage:\
    kmeans <data_file> <k> <iterations> --ALGORITHM\
Possible ALGORITHM values:\
    " << ATOMIC_ALG << "\
    " << ATOMIC_BIG_ALG << "\
    " << ATOMIC_TRANSPOSED_ALG << "\
	" << FEW_MEANS_ALG << "\
	" << FEW_MEANS_V2_ALG << "\
	" << FEW_MEANS_V3_ALG << "\
	" << MANY_DIMS_ALG << "\
	" << SIMPLE_ALG << "\
	" << WARP_PER_MEAN_ALG << "\
	" << WARP_PER_POINT_ALG << "\
	" << std::endl;
}

int main(int argc, const char* argv[])
{
	if (argc >= 6 && argc <= 8)
	{
		std::string file_name(argv[1]);
		std::string means_file_name = file_name;//(argv[3]);
		std::string clusters_file_name = file_name;// (argv[4]);
		int dataPos = file_name.find_last_of("/") + 1;
		means_file_name.erase(0, dataPos + std::string("data").length());
		means_file_name.insert(0, "means");
		clusters_file_name.erase(0, dataPos + std::string("data").length());
		clusters_file_name.insert(0, "cluster");
		std::string s_k(argv[2]);
		std::string s_iterations(argv[3]);
		uint32_t k = lexical_cast<uint32_t>(s_k);
		uint32_t iterations = lexical_cast<uint32_t>(s_iterations);
		int deviceID = lexical_cast<uint8_t>(argv[5]);
		if (verifyDeviceID(deviceID))
		{
			DEVICE_ID = deviceID;
		}
		else
		{
			throw std::runtime_error("Device ID not valid!");
		}

		uint64_t dataSize;
		uint64_t dimension;
		value_t* data = loadData(file_name, dataSize, dimension);
		value_t* means = (value_t*)calloc(k * dimension, sizeof(value_t));
		uint32_t* assignedClusters = (uint32_t*)calloc(dataSize * dimension, sizeof(uint32_t));
		memcpy(means, data, k * dimension * sizeof(value_t));

		// Add vectors in parallel.

		cudaError_t cudaStatus = cudaErrorInvalidConfiguration;
		if (argv[4] == ATOMIC_ALG)
			cudaStatus = countKMeansAtomic(iterations, dataSize, data, k, means, assignedClusters, dimension);
		if (argv[4] == ATOMIC_BIG_ALG)
			cudaStatus = countKMeansBIGDataAtomic(iterations, dataSize, data, k, means, assignedClusters, dimension);
		if (argv[4] == ATOMIC_TRANSPOSED_ALG)
			cudaStatus = countKMeansAtomicTransposed(iterations, dataSize, data, k, means, assignedClusters, dimension);
		if (argv[4] == FEW_MEANS_ALG)
			cudaStatus = countKMeansFewMeans(iterations, dataSize, data, k, means, assignedClusters, dimension, lexical_cast<uint32_t>(argv[6]));
		if (argv[4] == FEW_MEANS_V2_ALG)
			cudaStatus = countKMeansFewMeansV2(iterations, dataSize, data, k, means, assignedClusters, dimension);
		if (argv[4] == FEW_MEANS_V3_ALG)
			cudaStatus = countKMeansFewMeansV3(iterations, dataSize, data, k, means, assignedClusters, dimension, lexical_cast<uint32_t>(argv[6]), lexical_cast<uint32_t>(argv[7]));
		if (argv[4] == MANY_DIMS_ALG)
			cudaStatus = countKMeansManyDims(iterations, dataSize, data, k, means, assignedClusters, dimension);
		if (argv[4] == SIMPLE_ALG)
			cudaStatus = countKMeansSimple(iterations, dataSize, data, k, means, assignedClusters, dimension);
		if (argv[4] == WARP_PER_MEAN_ALG)
			cudaStatus = countKMeansWarpPerMean(iterations, dataSize, data, k, means, assignedClusters, dimension);
		if (argv[4] == WARP_PER_POINT_ALG)
			cudaStatus = countKMeansWarpPerPoint(iterations, dataSize, data, k, means, assignedClusters, dimension);

		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		saveResults(means_file_name, clusters_file_name, k, means, dataSize, data, assignedClusters, dimension);

		free(data);
		free(means);
		free(assignedClusters);

		return 0;
	}
	usage();
	return 1;
}

#pragma endregion