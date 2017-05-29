#ifndef HELPERS_H
#define HELPERS_H

#include "cudaException.h"
#include "cuda_runtime.h"

#include <stdint.h>
#include <sstream>
#include <vector>

typedef float value_t;
typedef unsigned char cluster_t;

static int DEVICE_ID;

template <typename T>
T lexical_cast(const std::string& x)
{
	std::istringstream stream(x);
	T res;
	stream >> res;
	return res;
}


value_t* loadData(const std::string& file_name, uint64_t& dataSize, uint64_t& dimension);

void saveResults(const std::string& means_file_name, const std::string& clusters_file_name, const uint32_t meansSize, const value_t* means, const uint32_t dataSize, const value_t* data, const uint32_t* assignedClusters, uint64_t& dimension);

bool verifyDeviceID(int deviceID);

void setDevice(int deviceID);

void allocateMemory(void** devicePointer, size_t size);

void allocateAndSetMemory(void** devicePointer, size_t size, int value);

void copyMemory(void *destination, const void *source, size_t count, enum cudaMemcpyKind kind);

void synchronizeDevice();

cudaError_t checkErrors();

template <typename T>
void transposeInput(T *arrayToTranspopse, uint32_t size, uint32_t dimension)
{
	uint32_t totalSize = size*dimension - 1;
	T temp; // holds element to be replaced, eventually becomes next element to move
	uint32_t next; // location of 't' to be moved
	uint32_t cycleBegin; // holds start of cycle
	uint32_t i; // iterator

	std::vector<bool> visited(totalSize + 1); // hash to mark moved elements

	visited[0] = true;
	visited[totalSize] = true;
	i = 1; // Note that A[0] and A[size-1] won't move
	while (i < totalSize)
	{
		cycleBegin = i;
		temp = arrayToTranspopse[i];
		do
		{
			// Input matrix [r x c]
			// Output matrix 
			// i_new = (i*r)%(N-1)
			next = (i*size) % totalSize;
			std::swap(arrayToTranspopse[next], temp);
			visited[i] = true;
			i = next;
		} while (i != cycleBegin);

		// Get Next Move (what about querying random location?)
		for (i = 1; i < totalSize && visited[i]; i++)
			;
	}
}

#endif //HELPERS_H