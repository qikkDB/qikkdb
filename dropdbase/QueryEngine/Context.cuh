#ifndef CONTEXT_H
#define CONTEXT_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include <memory>

#include "QueryEngineError.h"

class EngineCore;

class Context {
private:
	// Limit on the grid size
	static constexpr int32_t DEFAULT_GRID_DIMENSION_LIMIT = 65535;

	// Default grid dimensions for kernel launching
	static constexpr int32_t DEFAULT_GRID_DIMENSION = 65535;
	static constexpr int32_t DEFAULT_BLOCK_DIMENSION = 1024;

	// Number of opitimal threads per block queried for a specific GPU - currently bound to the context
	int32_t queried_block_dimension_;

	// Engine core for operations over the context - GPU only
	std::unique_ptr<EngineCore> engineCore_;

	// Registry for holding the last error
	QueryEngineError lastError_;

	// The currently bound device and found devices and their metadata
	int32_t boundDeviceID_;
	cudaDeviceProp boundDevice_;
	std::vector<cudaDeviceProp> devicesMetaInfo_;

	// Meyer's singleton
	Context(std::unique_ptr<EngineCore> engineCore)
		: engineCore_(std::move(engineCore)) {}
	~Context() = default;
	Context(const Context&) = delete;
	Context& operator=(const Context&) = delete;

public:

	// Get class instance, if class was not initialized prior a GPU instance is returned
	static Context& getInstance();

	// Get the engine core interface for kernel operations
	const EngineCore& getEngineCore() const { return *engineCore_; }

	// Get the last cuda error
	QueryEngineError& getLastError() { return lastError_; }

	// Operations on the grid dimensions
	int32_t calcGridDim(int32_t threadCount)
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

	void bindDeviceToContext(int32_t deviceID)
	{
		// TODO
	}

};

#endif 

