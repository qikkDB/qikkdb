#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <unordered_set>
#include <vector>

#include <memory>

#include "../Configuration.h"
#include "GPUError.h"
#include "CudaMemAllocator.h"
#include "GPUCore/GPUWhereInterpreter.h"
#include "GPUWhereFunctions.h"
#include "GPUMemoryCache.h"
#include "GPUError.h"
#include "../DataType.h"
class Database;

class Context
{
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
    std::vector<std::unique_ptr<GpuVMFunction[]>> gpuDispatchTables;

    // List of loaded databases
    std::unordered_map<std::string, std::shared_ptr<Database>> loadedDatabases_;

    // Meyer's singleton
	Context();

	~Context();

    Context(const Context&) = delete;

    Context& operator=(const Context&) = delete;

public:
    // Default device
    static const int32_t DEFAULT_DEVICE_ID = 0;

    // Get class instance, if class was not initialized prior a GPU instance is returned
	static Context& getInstance();

	// Operations on the grid dimensions
	int32_t calcGridDim(int32_t dataElementCount);

	// Get default block dimension
	const int32_t getBlockDim();

	// Get currently bound device
	const int32_t getBoundDeviceID();

	// Get found device count
	const int32_t getDeviceCount();

	// Querying info about devices and rebinding devices to the context
	const std::vector<cudaDeviceProp>& getDevicesMetaInfoList();

	// Bind device to context if neccessary, if id is out of range, bind the default device
	void bindDeviceToContext(int32_t deviceID);

	// Allocator methods
	CudaMemAllocator& GetAllocatorForDevice(int32_t deviceID);

	CudaMemAllocator& GetAllocatorForCurrentDevice();

	// Cache methods
	GPUMemoryCache& getCacheForDevice(int32_t deviceID);

	GPUMemoryCache& getCacheForCurrentDevice();

	GpuVMFunction* getDispatchTableForDevice(int32_t deviceID);

	GpuVMFunction* getDispatchTablesForCurrentDevice();

	std::unordered_map<std::string, std::shared_ptr<Database>>& GetLoadedDatabases();
};
