#include "atomicTasks.cuh"
#include "fewMeansTasks.cuh"
#include "manyDimensionsTasks.cuh"
#include "simpleTasks.cuh"

#include <stdint.h>
#include <stdexcept>
#include <stdio.h> 
#include <sstream>
#include <iostream>

#include <time.h>

#include <stdlib.h>
#include <vector>

uint64_t dimension;
typedef float value_t;
typedef unsigned char cluster_t;

#pragma region Common Functions
void usage()
{
    std::cout << "Usage:" << std::endl << "kmeans <data_file> <means_file> <clusters_file> <k> <iterations>" << std::endl << "kmeans --generate <data_file> <size> <seed>" << std::endl;
}

value_t* load(const std::string& file_name, uint64_t& dataSize)
{
	FILE* f = fopen(file_name.c_str(), "rb");
	if (!f) throw std::runtime_error("cannot open file for reading");
	//if (fseek(f, 0, SEEK_END)) throw std::runtime_error("seeking failed");
	if (!fread(&dataSize, sizeof(uint64_t), 1, f))  throw std::runtime_error("size cannot be read");
	if (!fread(&dimension, sizeof(uint64_t), 1, f))  throw std::runtime_error("dimension cannot be read");
	value_t* data = (value_t*)calloc(dataSize * dimension, sizeof(value_t));
	if (!fread(data, sizeof(value_t), dataSize * dimension, f))  throw std::runtime_error("value cannot be read");
	return data;
}

template<typename T>
T lexical_cast(const std::string& x)
{
	std::istringstream stream(x);
	T res;
	stream >> res;
	return res;
}

void save_results(const std::string& means_file_name, const std::string& clusters_file_name, const uint32_t meansSize, const value_t* means, const uint32_t dataSize, const value_t* data, const uint32_t* assignedClusters)
{
	FILE* f = fopen(means_file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	if (!fwrite(&dimension, sizeof(uint64_t), 1, f)) throw std::runtime_error("dimension cannot be written");
	//if (!fwrite(means, sizeof(value_t), dimension * meansSize, f)) throw std::runtime_error("value cannot be written");
	for (size_t i = 0; i < meansSize; i++)
	{
		if (!fwrite(&means[i*dimension], sizeof(value_t), dimension, f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&i, sizeof(unsigned char), 1, f)) throw std::runtime_error("value cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");


	f = fopen(clusters_file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	if (!fwrite(&dimension, sizeof(uint64_t), 1, f)) throw std::runtime_error("dimension cannot be written");
	for (size_t i = 0; i < dataSize; i++)
	{
		if (!fwrite(&data[i*dimension], sizeof(value_t), dimension, f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&assignedClusters[i], sizeof(unsigned char), 1, f)) throw std::runtime_error("value cannot be written");
		//if (!fwrite(&i, sizeof(value_t), 1, f)) throw std::runtime_error("distance cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");
}

int main(int argc, const char* argv[])
{
	if (argc == 6)
	{
		std::string file_name(argv[1]);
		std::string means_file_name = file_name;//(argv[2]);
		std::string clusters_file_name = file_name;// (argv[3]);
		int dataPos = file_name.find_last_of("/") + 1;
		means_file_name.erase(0, dataPos + std::string("data").length());
		means_file_name.insert(0, "means");
		clusters_file_name.erase(0, dataPos + std::string("data").length());
		clusters_file_name.insert(0, "cluster");
		std::string s_k(argv[4]);
		std::string s_iterations(argv[5]);
		uint32_t k = lexical_cast<uint32_t>(s_k);
		uint32_t iterations = lexical_cast<uint32_t>(s_iterations);
		uint64_t dataSize;

		value_t* data = load(file_name, dataSize);
		value_t* means = (value_t*)calloc(k * dimension, sizeof(value_t));
		uint32_t* assignedClusters = (uint32_t*)calloc(dataSize * dimension, sizeof(uint32_t));
		memcpy(means, data, k * dimension * sizeof(value_t));

		// Add vectors in parallel.
		cudaError_t cudaStatus = countKMeansSimple(iterations, dataSize, data, k, means, assignedClusters, dimension);
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

		save_results(means_file_name, clusters_file_name, k, means, dataSize, data, assignedClusters);

		free(data);
		free(means);
		free(assignedClusters);

		return 0;
	}
	usage();
	return 1;
}

#pragma endregion