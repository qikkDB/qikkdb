#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <unordered_set>
#include <vector>

#include <memory>

#include "../Configuration.h"
#include "CudaMemAllocator.h"
#include "GPUError.h"
#include "GPUMemoryCache.h"

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

    // Meyer's singleton
    Context()
    {
        // Save found device count and notify the user
        if (cudaGetDeviceCount(&deviceCount_) != cudaSuccess)
        {
            throw std::invalid_argument("INFO: Unable to get device count");
        }

        // DANGER     DANGER     DANGER     DANGER      DANGER      DANGER 
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        /////////////////////// DEADLY DEADLY DEADLY /////////////////////// 
        //deviceCount_ = 1;
        /////////////////////// DEADLY DEADLY DEADLY /////////////////////// 

        printf("INFO: Found %d CUDA devices\n", deviceCount_);
        const int cachePercentage = Configuration::GetInstance().GetGPUCachePercentage();
        // Get devices information
        for (int32_t i = 0; i < deviceCount_; i++)
        {
            // Bind device and initialize everything for a device allocators/cache
            bindDeviceToContext(i);

            // Get devices information
            cudaDeviceProp deviceProp;
            if (cudaGetDeviceProperties(&deviceProp, i) != cudaSuccess)
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
            gpuCaches_.emplace_back(std::make_unique<GPUMemoryCache>(i, cacheSize));

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
                if (cudaDeviceCanAccessPeer(&canAccessPeer, DEFAULT_DEVICE_ID, i) != cudaSuccess)
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

    ~Context()
    {
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
    /// The default bound CUDA device ID
    static const int32_t DEFAULT_DEVICE_ID = 0;

    /// Get a class instance, if class was not initialized prior a new instance is returned
    /// <returns>an instance of this class</returns>
    static Context& getInstance()
    {
        // Static instance - constructor called only once
        static Context instance;
        return instance;
    }

    /// Calculate the size of the CUDA grid for kernel launches
    /// <param name="dataElementCount">number of data elements for CUDA to process</param>
    int32_t calcGridDim(size_t dataElementCount)
    {
        if (dataElementCount <= 0)
        {
            CheckQueryEngineError(QueryEngineErrorType::GPU_EXTENSION_ERROR,
                                  "Data Element Count must be > 0");
        }
        int blockCount = (dataElementCount + getBlockDim() - 1) / getBlockDim();
        if (blockCount >= (DEFAULT_GRID_DIMENSION_LIMIT + 1))
        {
            blockCount = DEFAULT_GRID_DIMENSION_LIMIT;
        }
        return blockCount;
    }

    /// Get default block dimension - the size of a stream multiprocessor
    /// <returns>the size of an optimal block</returns>
    int32_t getBlockDim() const
    {
        return queriedBlockDimensionList[getBoundDeviceID()];
    }

    /// Get the currently bound device to the context
    /// <returns>the bound device ID</returns>
     int32_t getBoundDeviceID() const
    {
        int boundDeviceID;
        cudaGetDevice(&boundDeviceID);
        return boundDeviceID;
    }

    /// Get the number of found devices on a platform
    /// <returns>the number of found devices</returns>
    int32_t getDeviceCount() const
    {
        return deviceCount_;
    }

    /// Query info about devices and rebinding devices to the context
    /// <returns>Returns a vector of structures of type cudaDeviceProp, the device properties obtainable from
	/// this structre are documented in the CUDA documentation</returns>
    const std::vector<cudaDeviceProp>& getDevicesMetaInfoList() const
    {
        return devicesMetaInfoList_;
    }

    /// Bind device to the context if neccessary, if the given id is out of range,
	/// the default device is bound
    /// <param name="deviceID">the ID of the device to be bound</param>
    void bindDeviceToContext(int32_t deviceID)
    {
        // Check for invalid range
        if (deviceID < 0 || deviceID >= deviceCount_)
        {
            throw std::out_of_range("ERROR: Device ID not present");
        }

        cudaSetDevice(deviceID);
    }

    /// Obtain the memory allocator for a given device
    /// <param name="deviceID">the ID of the device whose allocator has to be found</param>
    /// <returns>the memory allocator for memory operations on the selected device</returns>
    CudaMemAllocator& GetAllocatorForDevice(int32_t deviceID)
    {
        // Check for invalid range
        if (deviceID < 0 || deviceID >= deviceCount_)
        {
            throw std::out_of_range("ERROR: Device ID not present");
        }

        return *gpuAllocators_.at(deviceID);
    }

	/// Obtain the memory allocator for the currently bound device
    /// <returns>the memory allocator for memory operations on the current device</returns>
    CudaMemAllocator& GetAllocatorForCurrentDevice()
    {
        return *gpuAllocators_.at(getBoundDeviceID());
    }

    /// Obtain the cache for a given device
    /// <param name="deviceID">the ID of the device whose cache has to be found</param>
    /// <returns>the gpu cache on the selected device</returns>
    GPUMemoryCache& getCacheForDevice(int32_t deviceID)
    {
        // Check for invalid range
        if (deviceID < 0 || deviceID >= deviceCount_)
        {
            throw std::out_of_range("ERROR: Device ID not present");
        }

        return *gpuCaches_.at(deviceID);
    }

	/// Obtain the cache for the currently bound device
    /// <returns>the cache on the current device</returns>
    GPUMemoryCache& getCacheForCurrentDevice()
    {
        return *gpuCaches_.at(getBoundDeviceID());
    }

    std::unordered_map<std::string, std::shared_ptr<Database>>& GetLoadedDatabases()
    {
        return loadedDatabases_;
    }
};
