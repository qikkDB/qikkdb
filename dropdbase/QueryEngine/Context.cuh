#ifndef CONTEXT_H
#define CONTEXT_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include "EngineCore.cuh"
#include "QueryEngineError.h"

class Context {
private:
	// Limit on the grid size
	static constexpr int32_t DEFAULT_GRID_DIMENSION_LIMIT = 65535;

	// Default grid dimensions for kernel launching
	static constexpr int32_t DEFAULT_GRID_DIMENSION = 65535;
	static constexpr int32_t DEFAULT_BLOCK_DIMENSION = 1024;

	// Number of opitimal threads per block queried for a specific GPU - currently bound to the context
	int32_t queried_block_dimension;

	// Engine core interface for operations over the context
	EngineCore engineCore;

	// Registry for holding the last error
	QueryEngineError lastError;

	// The currently bound device and found devices and their metadata
	int32_t boundDeviceID;
	cudaDeviceProp boundDevice;
	std::vector<cudaDeviceProp> devicesMetaInfo;

	// Meyer's singleton
	Context(EngineCore::Device device) : engineCore(device) {};
	~Context() {}
	Context(const Context&) = delete;
	Context& operator=(const Context&) = delete;

	static Context& getInnerInstance(EngineCore::Device device = EngineCore::GPU)
	{
		// Static instance - constructor called only once
		static Context instance(device);
		return instance;
	}

public:
	// Initialize class
	static void initGPUContext() { getInnerInstance(EngineCore::GPU); }
	static void initCPUContext() { getInnerInstance(EngineCore::CPU); }

	// Get class instance, if class was not initialized prior a GPU instance is returned
	static Context& getInstance() { return getInnerInstance(); }

	// Get the engine core interface for kernel operations
	const EngineCore& getEngineCore() const { return engineCore; }

	// Get the last cuda error
	const QueryEngineError& getLastError() const { return lastError; }

	// Operations on the grid dimensions
	int32_t calcGridDim(int32_t threadCount)
	{
		int blockCount = (threadCount + queried_block_dimension - 1) / queried_block_dimension;
		if (blockCount >= (DEFAULT_GRID_DIMENSION_LIMIT + 1))
		{
			blockCount = DEFAULT_GRID_DIMENSION_LIMIT;
		}
		return blockCount;
	}

	int32_t getBlockDim() { return DEFAULT_BLOCK_DIMENSION; }

	// Querying info about devices and rebinding devices to the context
	const std::vector<cudaDeviceProp>& getDevicesMetaInfoList() const { return devicesMetaInfo; }

	void bindDeviceToContext(int32_t deviceID)
	{
		// TODO
	}

};

#endif 

