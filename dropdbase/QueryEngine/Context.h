#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <unordered_set>

#include <memory>

#include "QueryEngineError.h"
#include "CudaMemAllocator.h"
#include "GPUMemoryCache.h"
#include "../Configuration.h"

class Database;

class Context {
private:

	// Limit on the grid size
	static const int32_t DEFAULT_GRID_DIMENSION_LIMIT = 65535;

	// Default grid dimensions for kernel launching
	static const int32_t DEFAULT_GRID_DIMENSION = 65535;
	static const int32_t DEFAULT_BLOCK_DIMENSION = 1024;

	// Number of opitimal threads per block queried for a specific GPU - currently bound to the context
	std::vector<int32_t> queriedBlockDimensionList;

	// The found devices and their metadata
	int32_t deviceCount_;
	std::vector<cudaDeviceProp> devicesMetaInfoList_;

    // Move cannot be implemented for allocator and cache, they keep iterators to internal vectors
    std::vector<std::unique_ptr<CudaMemAllocator>> gpuAllocators_;
    std::vector<std::unique_ptr<GPUMemoryCache>> gpuCaches_;

	// List of loaded databases
	std::unordered_map<std::string, std::shared_ptr<Database>> loadedDatabases_;

	// Meyer's singleton
	Context()
	{
		// Save found device count and notify the user
		if (cudaGetDeviceCount(&deviceCount_) != CUDA_SUCCESS)
		{
			throw std::invalid_argument("INFO: Unable to get device count");
		}
		printf("INFO: Found %d CUDA devices\n", deviceCount_);
		const int cachePercentage = Configuration::GetInstance().GetGPUCachePercentage();
		// Get devices information
		for (int32_t i = 0; i < deviceCount_; i++)
		{
			// Bind device and initialize everything for a device allocators/cache
			bindDeviceToContext(i);

			// Get devices information
			cudaDeviceProp deviceProp;
			if (cudaGetDeviceProperties(&deviceProp, i) != CUDA_SUCCESS)
			{
				throw std::invalid_argument("ERROR: Failed to get GPU info");
			}
			devicesMetaInfoList_.push_back(deviceProp);
			// Print memory info
			size_t free, total;
			cudaMemGetInfo(&free, &total);

			// Initialize allocators
			gpuAllocators_.emplace_back(std::make_unique<CudaMemAllocator>(i));

			// Initialize cache
			size_t cacheSize = static_cast<int64_t>(free * static_cast<double>(cachePercentage) / 100.0);
			gpuCaches_.emplace_back(std::make_unique<GPUMemoryCache>(i,cacheSize));

			// Get the correct blockDim from the device - use always based on the bound device - optimal for kernels
			queriedBlockDimensionList.push_back(deviceProp.maxThreadsPerBlock);

			// Print device info
			printf("INFO: Device ID: %d: %s \t maxBlockDim: %d\n", i, deviceProp.name, deviceProp.maxThreadsPerBlock);

			printf("INFO: Memory: Total: %zu B Free: %zu B Cache: %zu B\n", total, free, cacheSize);
		}

		// Bind default device and notify the user
		bindDeviceToContext(DEFAULT_DEVICE_ID);
		printf("INFO: Bound default device ID: %d\n", getBoundDeviceID());

		// Enable peer to peer communication of each GPU to the default device
		// This operation is unidirectional
		for (int32_t i = 0; i < deviceCount_; i++)
		{
			if (i != DEFAULT_DEVICE_ID)
			{
				int32_t canAccessPeer;
				if (cudaDeviceCanAccessPeer(&canAccessPeer, DEFAULT_DEVICE_ID, i) != CUDA_SUCCESS)
				{
					throw std::invalid_argument("ERROR: CUDA peer acces not supported");
				}
				if (canAccessPeer == 1)
				{
					cudaDeviceEnablePeerAccess(i, DEFAULT_DEVICE_ID);
				}
			}
		}
	};

	~Context() {
		for (int32_t i = 0; i < deviceCount_; i++)
		{
			// Bind device and clean up
			bindDeviceToContext(i);
			cudaDeviceReset();
		}
	}

	Context(const Context&) = delete;

	Context& operator=(const Context&) = delete;

public:
	// Default device
	static const int32_t DEFAULT_DEVICE_ID = 0;

	// Get class instance, if class was not initialized prior a GPU instance is returned
	static Context& getInstance()
	{
		// Static instance - constructor called only once
		static Context instance;
		return instance;
	}

	// Operations on the grid dimensions
	int32_t calcGridDim(int32_t dataElementCount)
	{
		int blockCount = (dataElementCount + getBlockDim() - 1) / getBlockDim();
		if (blockCount >= (DEFAULT_GRID_DIMENSION_LIMIT + 1))
		{
			blockCount = DEFAULT_GRID_DIMENSION_LIMIT;
		}
		return blockCount;
	}

	// Get default block dimension
	const int32_t getBlockDim() { 
		return queriedBlockDimensionList[getBoundDeviceID()];
	}

	// Get currently bound device
	const int32_t getBoundDeviceID() { 
		int boundDeviceID;
		cudaGetDevice(&boundDeviceID);
		return boundDeviceID;
	}

	// Get found device count
	const int32_t getDeviceCount() { 
		return deviceCount_; 
	}

	// Querying info about devices and rebinding devices to the context
	const std::vector<cudaDeviceProp>& getDevicesMetaInfoList() { 
		return devicesMetaInfoList_; 
	}

	// Bind device to context if neccessary, if id is out of range, bind the default device
	void bindDeviceToContext(int32_t deviceID) {
		//Check for invalid range
		if (deviceID < 0 || deviceID >= deviceCount_)
		{
			throw std::out_of_range("ERROR: Device ID not present");
		}

		cudaSetDevice(deviceID);
	}

	// Allocator methods
	CudaMemAllocator& GetAllocatorForDevice(int32_t deviceID) {
		//Check for invalid range
		if (deviceID < 0 || deviceID >= deviceCount_)
		{
			throw std::out_of_range("ERROR: Device ID not present");
		}

		return *gpuAllocators_.at(deviceID);
	}

	CudaMemAllocator& GetAllocatorForCurrentDevice()
	{
		return *gpuAllocators_.at(getBoundDeviceID());
	}

	// Cache methods
	GPUMemoryCache& getCacheForDevice(int32_t deviceID) { 
		//Check for invalid range
		if (deviceID < 0 || deviceID >= deviceCount_)
		{
			throw std::out_of_range("ERROR: Device ID not present");
		}

		return *gpuCaches_.at(deviceID); 
	}

	GPUMemoryCache& getCacheForCurrentDevice()
	{
		return *gpuCaches_.at(getBoundDeviceID());
	}

	std::unordered_map<std::string, std::shared_ptr<Database>>& GetLoadedDatabases()
	{
		return loadedDatabases_;
	}
};

