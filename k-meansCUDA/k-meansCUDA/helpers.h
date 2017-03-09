#ifndef HELPERS_H
#define HELPERS_H

#include "cudaException.h"
#include "cuda_runtime.h"

void setDevice(int deviceID);

void allocateMemory(void** devicePointer, size_t size);

void allocateAndSetMemory(void** devicePointer, size_t size, int value);

void copyMemory(void *destination, const void *source, size_t count, enum cudaMemcpyKind kind);

void synchronizeDevice();

cudaError_t checkErrors();

#endif //HELPERS_H