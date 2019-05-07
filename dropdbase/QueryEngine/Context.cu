#include "Context.h"
#include "GPUError.h"

Context::Context()
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
		gpuCaches_.emplace_back(std::make_unique<GPUMemoryCache>(i, cacheSize));

		// Get the correct blockDim from the device - use always based on the bound device - optimal for kernels
		queriedBlockDimensionList.push_back(deviceProp.maxThreadsPerBlock);
		const int32_t DISPATCH_ARRAY_SIZE = DataType::COLUMN_INT * DataType::COLUMN_INT * GPUWhereFunctions::FUNC_COUNT;
		std::unique_ptr<GpuVMFunction[]> dispatchTable(new GpuVMFunction[DISPATCH_ARRAY_SIZE]);
		CheckCudaError(cudaHostRegister(dispatchTable.get(), DISPATCH_ARRAY_SIZE * sizeof(GpuVMFunction),
			cudaHostRegisterDefault));
		GpuVMFunction* gpuDispatchPtr;
		CheckCudaError(cudaHostGetDevicePointer(&gpuDispatchPtr, dispatchTable.get(), 0));

		kernel_fill_gpu_dispatch_table << <calcGridDim(DISPATCH_ARRAY_SIZE), getBlockDim() >> > (gpuDispatchPtr, DISPATCH_ARRAY_SIZE);
		CheckCudaError(cudaHostUnregister(dispatchTable.get()));

		gpuDispatchTables.emplace_back(std::move(dispatchTable));

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

Context::~Context()
{
	for (int32_t i = 0; i < deviceCount_; i++)
	{
		// Bind device and clean up
		bindDeviceToContext(i);
		cudaDeviceReset();
	}
}

Context& Context::getInstance()
{
	// Static instance - constructor called only once
	static Context instance;
	return instance;
}

// Operations on the grid dimensions
int32_t Context::calcGridDim(int32_t dataElementCount)
{
	int blockCount = (dataElementCount + getBlockDim() - 1) / getBlockDim();
	if (blockCount >= (DEFAULT_GRID_DIMENSION_LIMIT + 1))
	{
		blockCount = DEFAULT_GRID_DIMENSION_LIMIT;
	}
	return blockCount;
}

// Get default block dimension
const int32_t Context::getBlockDim()
{
	return queriedBlockDimensionList[getBoundDeviceID()];
}

// Get currently bound device
const int32_t Context::getBoundDeviceID()
{
	int boundDeviceID;
	cudaGetDevice(&boundDeviceID);
	return boundDeviceID;
}

// Get found device count
const int32_t Context::getDeviceCount()
{
	return deviceCount_;
}

// Querying info about devices and rebinding devices to the context
const std::vector<cudaDeviceProp>& Context::getDevicesMetaInfoList()
{
	return devicesMetaInfoList_;
}

// Bind device to context if neccessary, if id is out of range, bind the default device
void Context::bindDeviceToContext(int32_t deviceID)
{
	// Check for invalid range
	if (deviceID < 0 || deviceID >= deviceCount_)
	{
		throw std::out_of_range("ERROR: Device ID not present");
	}

	cudaSetDevice(deviceID);
}

// Allocator methods
CudaMemAllocator& Context::GetAllocatorForDevice(int32_t deviceID)
{
	// Check for invalid range
	if (deviceID < 0 || deviceID >= deviceCount_)
	{
		throw std::out_of_range("ERROR: Device ID not present");
	}

	return *gpuAllocators_.at(deviceID);
}

CudaMemAllocator& Context::GetAllocatorForCurrentDevice()
{
	return *gpuAllocators_.at(getBoundDeviceID());
}

// Cache methods
GPUMemoryCache& Context::getCacheForDevice(int32_t deviceID)
{
	// Check for invalid range
	if (deviceID < 0 || deviceID >= deviceCount_)
	{
		throw std::out_of_range("ERROR: Device ID not present");
	}

	return *gpuCaches_.at(deviceID);
}

GPUMemoryCache& Context::getCacheForCurrentDevice()
{
	return *gpuCaches_.at(getBoundDeviceID());
}

GpuVMFunction* Context::getDispatchTableForDevice(int32_t deviceID)
{
	return gpuDispatchTables.at(deviceID).get();
}

GpuVMFunction* Context::getDispatchTableForCurrentDevice()
{
	return gpuDispatchTables.at(getBoundDeviceID()).get();
}

std::unordered_map<std::string, std::shared_ptr<Database>>& Context::GetLoadedDatabases()
{
	return loadedDatabases_;
}