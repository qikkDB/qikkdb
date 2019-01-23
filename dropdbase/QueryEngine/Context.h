#ifndef CONTEXT_H
#define CONTEXT_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include <memory>

#include "QueryEngineError.h"

class Context {
private:
	// Limit on the grid size
	static const int32_t DEFAULT_GRID_DIMENSION_LIMIT = 65535;

	// Default grid dimensions for kernel launching
	static const int32_t DEFAULT_GRID_DIMENSION = 65535;
	static const int32_t DEFAULT_BLOCK_DIMENSION = 1024;

	// Number of opitimal threads per block queried for a specific GPU - currently bound to the context
	int32_t queried_block_dimension_;

	// Registry for holding the last error
	QueryEngineError lastError_;

	// The currently bound device and found devices and their metadata
	int32_t boundDeviceID_;
	cudaDeviceProp boundDevice_;
	std::vector<cudaDeviceProp> devicesMetaInfo_;

	// Meyer's singleton
	Context() : queried_block_dimension_(DEFAULT_BLOCK_DIMENSION) {};
	~Context() = default;
	Context(const Context&) = delete;
	Context& operator=(const Context&) = delete;

public:
	// Get class instance, if class was not initialized prior a GPU instance is returned
	static Context& Context::getInstance()
	{
		// Static instance - constructor called only once
		static Context instance;
		return instance;
	}

	// Get the last cuda error
	QueryEngineError& getLastError() { return lastError_; }

	// Operations on the grid dimensions
	int32_t Context::calcGridDim(int32_t threadCount)
	{
		int blockCount = (threadCount + queried_block_dimension_ - 1) / queried_block_dimension_;
		if (blockCount >= (DEFAULT_GRID_DIMENSION_LIMIT + 1))
		{
			blockCount = DEFAULT_GRID_DIMENSION_LIMIT;
		}
		return blockCount;
	}

	int32_t getBlockDim() { return DEFAULT_BLOCK_DIMENSION; }

	// Querying info about devices and rebinding devices to the context
	const std::vector<cudaDeviceProp>& getDevicesMetaInfoList() const { return devicesMetaInfo_; }

	// Bind device to context if neccessary
	void bindDeviceToContext(int32_t deviceID) { }

};

#endif 

