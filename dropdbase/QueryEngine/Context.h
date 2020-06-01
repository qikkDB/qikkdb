#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <unordered_set>
#include <vector>

#include <memory>

#include "CudaMemAllocator.h"
#include "GPUError.h"
#include "GPUMemoryCache.h"
#include "../Configuration.h"


class Database;

/// This class encapsulates the contex or state of the whole CUDA api in this program as an abstract state machine
/// This class isused for obtainign CUDA device meta information for kernel launches and device switching
/// This class utilizes Meyer's singleton for accesibility troughout the program
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

    // List of loaded databases
    std::unordered_map<std::string, std::shared_ptr<Database>> loadedDatabases_;

    // Community limitations. These values might be loaded according to licence in the future.
    const int64_t rowsLimit_ = 1000000000;
    const int32_t columnsLimit_ = 8;
    const int32_t tablesLimit_ = 4;
    const int32_t databasesLimit_ = 2;
    const int32_t gpusLimit_ = 1;

    // Meyer's singleton
    Context();

    ~Context();

    Context(const Context&) = delete;

    Context& operator=(const Context&) = delete;

    void Initialize();

public:
    /// The default bound CUDA device ID
    static const int32_t DEFAULT_DEVICE_ID = 0;

    /// Get a class instance, if class was not initialized prior a new instance is returned
    /// <returns>an instance of this class</returns>
    static Context& getInstance();

    /// Calculate the size of the CUDA grid for kernel launches
    /// <param name="dataElementCount">number of data elements for CUDA to process</param>
    int32_t calcGridDim(size_t dataElementCount);

    /// Get default block dimension - the size of a stream multiprocessor
    /// <returns>the size of an optimal block</returns>
    int32_t getBlockDim() const;

    /// Get default block dimension for a polygon operation - half the size of a stream
    /// multiprocessor <returns>the size of an optimal block</returns>
    int32_t getBlockDimPoly() const;

    /// Get the currently bound device to the context
    /// <returns>the bound device ID</returns>
    int32_t getBoundDeviceID() const;

    /// Get the number of found devices on a platform
    /// <returns>the number of found devices</returns>
    int32_t getDeviceCount() const;

    /// Query info about devices and rebinding devices to the context
    /// <returns>Returns a vector of structures of type cudaDeviceProp, the device properties
    /// obtainable from this structre are documented in the CUDA documentation</returns>
    const std::vector<cudaDeviceProp>& getDevicesMetaInfoList() const;

    /// Bind device to the context if neccessary, if the given id is out of range,
    /// the default device is bound
    /// <param name="deviceID">the ID of the device to be bound</param>
    void bindDeviceToContext(int32_t deviceID);

    /// Obtain the memory allocator for a given device
    /// <param name="deviceID">the ID of the device whose allocator has to be found</param>
    /// <returns>the memory allocator for memory operations on the selected device</returns>
    CudaMemAllocator& GetAllocatorForDevice(int32_t deviceID);

    /// Obtain the memory allocator for the currently bound device
    /// <returns>the memory allocator for memory operations on the current device</returns>
    CudaMemAllocator& GetAllocatorForCurrentDevice();

    /// Obtain the cache for a given device
    /// <param name="deviceID">the ID of the device whose cache has to be found</param>
    /// <returns>the gpu cache on the selected device</returns>
    GPUMemoryCache& getCacheForDevice(int32_t deviceID);

    /// Obtain the cache for the currently bound device
    /// <returns>the cache on the current device</returns>
    GPUMemoryCache& getCacheForCurrentDevice();

    std::unordered_map<std::string, std::shared_ptr<Database>>& GetLoadedDatabases();

    int64_t GetRowsLimit() const;

    int32_t GetColumnsLimit() const;

    int32_t GetTablesLimit() const;

    int32_t GetDatabasesLimit() const;

    int32_t GetGpusLimit() const;
};
