#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include <memory>

#include "QueryEngineError.h"
#include "CudaMemAllocator.h"

class Context {
private:
	// Limit on the grid size
	static const int32_t DEFAULT_GRID_DIMENSION_LIMIT = 65535;

	// Default grid dimensions for kernel launching
	static const int32_t DEFAULT_GRID_DIMENSION = 65535;
	static const int32_t DEFAULT_BLOCK_DIMENSION = 1024;

	// Number of opitimal threads per block queried for a specific GPU - currently bound to the context
	int32_t queried_block_dimension_;

	// The currently bound device and found devices and their metadata
	int32_t boundDeviceID_;
	cudaDeviceProp boundDevice_;
	std::vector<cudaDeviceProp> devicesMetaInfo_;
	// Move cannot be implemented for allocator, it keeps iterators to internal vectors
	std::vector<std::unique_ptr<CudaMemAllocator>> gpuAllocators_;
	// Meyer's singleton
	Context() : queried_block_dimension_(DEFAULT_BLOCK_DIMENSION) 
	{
		int devCount;
		if (cudaGetDeviceCount(&devCount) != CUDA_SUCCESS)
		{
			throw std::invalid_argument("Unable to get device count");
		}
		for (int i = 0; i < devCount; i++)
		{
			gpuAllocators_.emplace_back(std::make_unique<CudaMemAllocator>(i));
		}

		// TODO - Add device detection
	};
	~Context() = default;
	Context(const Context&) = delete;
	Context& operator=(const Context&) = delete;

public:
	// Get class instance, if class was not initialized prior a GPU instance is returned
	static Context& getInstance()
	{
		// Static instance - constructor called only once
		static Context instance;
		return instance;
	}

	// Operations on the grid dimensions
	int32_t calcGridDim(int32_t threadCount) const
	{
		int blockCount = (threadCount + queried_block_dimension_ - 1) / queried_block_dimension_;
		if (blockCount >= (DEFAULT_GRID_DIMENSION_LIMIT + 1))
		{
			blockCount = DEFAULT_GRID_DIMENSION_LIMIT;
		}
		return blockCount;
	}

	constexpr int32_t getBlockDim() const { return DEFAULT_BLOCK_DIMENSION; }

	// Querying info about devices and rebinding devices to the context
	const std::vector<cudaDeviceProp>& getDevicesMetaInfoList() const { return devicesMetaInfo_; }

	// Bind device to context if neccessary
	void bindDeviceToContext(int32_t deviceID) { }
	CudaMemAllocator& GetAllocatorForDevice(int32_t deviceID) { return *gpuAllocators_.at(deviceID); }
	CudaMemAllocator& GetAllocatorForCurrentDevice()
	{
		int deviceID;
		cudaGetDevice(&deviceID);
		return *gpuAllocators_.at(deviceID); 
	}

};

