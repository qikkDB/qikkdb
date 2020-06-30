#include "Context.h"

Context::Context()
{
    Initialize();
}

Context::~Context()
{
    gpuCaches_.clear();
    gpuAllocators_.clear();
    for (int32_t i = 0; i < deviceCount_; i++)
    {
        // Bind device and clean up
        bindDeviceToContext(i);
        cudaDeviceReset();
    }
}

void Context::Initialize()
{
    const cudaError_t err = cudaGetDeviceCount(&deviceCount_);
    if (err != cudaSuccess)
    {
        CudaLogBoost::getInstance(CudaLogBoost::error)
            << "cudaGetDeviceCount returns " << err << " which is " << cudaGetErrorName(err) << '\n';
        throw std::invalid_argument("ERROR: Unable to get device count");
    }

    // DANGER DANGER DANGER !!! DEADLY DEADLY DEADLY !!!
    // Don't touch anymore :D Use config UsingMultipleGPUs instead
    if (!Configuration::GetInstance().IsUsingMultipleGPUs())
    {
        deviceCount_ = 1;
    }
#ifdef COMMUNITY
    deviceCount_ = gpusLimit_;
#endif // COMMUNITY
    const int cachePercentage = Configuration::GetInstance().GetGPUCachePercentage();
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Initializing CUDA devices..." << '\n';
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Found " << deviceCount_ << " CUDA devices" << '\n';

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
        CudaLogBoost::getInstance(CudaLogBoost::info) << "Initializing memory for device " << i << "\n";
        // Initialize allocators
        gpuAllocators_.emplace_back(std::make_unique<CudaMemAllocator>(i));
        CudaLogBoost::getInstance(CudaLogBoost::info) << "Initializing cache for device " << i << "\n";
        // Initialize cache
        size_t cacheSize = static_cast<int64_t>(free * static_cast<double>(cachePercentage) / 100.0);
        gpuCaches_.emplace_back(std::make_unique<GPUMemoryCache>(i, cacheSize));

        // Get the correct blockDim from the device - use always based on the bound device - optimal for kernels
        queriedBlockDimensionList_.push_back(deviceProp.maxThreadsPerBlock);

        // Print device info
        CudaLogBoost::getInstance(CudaLogBoost::info)
            << "Device " << i << " Initialization done " << deviceProp.name
            << "\t maxBlockDim: " << deviceProp.maxThreadsPerBlock << "\n";

        CudaLogBoost::getInstance(CudaLogBoost::info) << "Memory: Total: " << total << " B Free: " << free
                                                      << "B Cache: " << cacheSize << " B" << '\n';
    }

    // Bind default device and notify the user
    bindDeviceToContext(DEFAULT_DEVICE_ID);
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Bound default device ID: " << getBoundDeviceID() << '\n';

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
}

Context& Context::getInstance()
{
    // Static instance - constructor called only once
    static Context instance;
    return instance;
}

int32_t Context::calcGridDim(size_t dataElementCount)
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

int32_t Context::getBlockDim() const
{
    return queriedBlockDimensionList_[getBoundDeviceID()];
}

int32_t Context::getBlockDimPoly() const
{
    return queriedBlockDimensionList_[getBoundDeviceID()] / 2;
}

int32_t Context::getBoundDeviceID() const
{
    int boundDeviceID = -1;
    cudaGetDevice(&boundDeviceID);
    return boundDeviceID;
}

int32_t Context::getDeviceCount() const
{
    return deviceCount_;
}

const std::vector<cudaDeviceProp>& Context::getDevicesMetaInfoList() const
{
    return devicesMetaInfoList_;
}

void Context::bindDeviceToContext(int32_t deviceID)
{
    // Check for invalid range
    if (deviceID < 0 || deviceID >= deviceCount_)
    {
        throw std::out_of_range("ERROR: Device ID not present");
    }
    cudaDeviceSynchronize();
    auto error = cudaSetDevice(deviceID);
    cudaDeviceSynchronize();
    CheckCudaError(error);
}

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

std::unordered_map<std::string, std::shared_ptr<Database>>& Context::GetLoadedDatabases()
{
    return loadedDatabases_;
}

void Context::CheckDatabasesLimit(const int64_t databasesCount) const
{
#ifdef COMMUNITY
    if (databasesCount >= databasesLimit_)
    {
        throw std::runtime_error(
            "Unable to insert new database. Community version supports only up to " +
            std::to_string(databasesLimit_) + " databases.");
    }
#endif // COMMUNITY
}

void Context::CheckTablesLimit(const int64_t tablesCount) const
{
#ifdef COMMUNITY
    if (tablesCount >= tablesLimit_)
    {
        throw std::runtime_error(
            "Unable to insert new table. Community version supports only up to " +
            std::to_string(tablesLimit_) + " tables.");
    }
#endif // COMMUNITY
}

void Context::CheckColumnsLimit(const int64_t columnsCount) const
{
#ifdef COMMUNITY
    if (columnsCount >= columnsLimit_)
    {
        throw std::runtime_error(
            "Unable to insert new column. Community version supports only up to " +
            std::to_string(columnsLimit_) + " columns.");
    }
#endif // COMMUNITY
}

void Context::CheckRowsLimit(const int64_t rowsCount) const
{
#ifdef COMMUNITY
    if (rowsCount >= rowsLimit_)
    {
        throw std::runtime_error(
            "Unable to insert new data. Community version supports only up to " +
            std::to_string(rowsLimit_) + " rows.");
    }
#endif // COMMUNITY
}