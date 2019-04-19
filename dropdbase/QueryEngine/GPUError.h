#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string>

/// <summary>
/// Enum for different QueryEngine errors
/// </summary>
enum QueryEngineErrorType
{
    GPU_EXTENSION_SUCCESS = 0, // Return code for successful operations
    GPU_EXTENSION_ERROR, // Return code for all CUDA errors
    GPU_DIVISION_BY_ZERO_ERROR, // Return code for division by zero
    GPU_INTEGER_OVERFLOW_ERROR, // Return code for integer overflow
    GPU_HASH_TABLE_FULL, // Return code for exceeding hash table limit (e.g. at group by with too many buckets)
    GPU_UNKNOWN_AGG_FUN, // The used function in the group by command is unknown
    GPU_NOT_FOUND_ERROR, // Return code for no detected GPU
    GPU_MEMORY_MAPPING_NOT_SUPPORTED_ERROR, // Return code for no memory mapping
    GPU_DRIVER_NOT_FOUND_EXCEPTION // Return code for not found nvidia driver
};

/// <summary>
/// Generic GPU Error
/// </summary>
class gpu_error : public std::runtime_error
{
public:
    explicit gpu_error(const std::string& what_arg) : runtime_error(what_arg)
    {
    }
    ~gpu_error()
    {
    }
};

/// <summary>
/// Error for CUDA "internal" errors
/// </summary>
class cuda_error : public gpu_error
{
private:
    cudaError_t cudaError_;

public:
    explicit cuda_error(cudaError_t cudaError)
    : gpu_error("CUDA Error " + std::to_string(static_cast<int32_t>(cudaError)) + ": " +
                std::string(cudaGetErrorName(cudaError)))
    {
        cudaError_ = cudaError;
    }

	~cuda_error()
	{
	}

    cudaError_t GetCudaError()
    {
        return cudaError_;
    }
};

/// <summary>
/// Error for our QueryEngine errors
/// </summary>
class query_engine_error : public gpu_error
{
private:
    QueryEngineErrorType gpuErrorType_;

public:
    explicit query_engine_error(QueryEngineErrorType gpuErrorType, const std::string& message)
    : gpu_error("GPU Error " + std::to_string(gpuErrorType) + (message.size() > 0 ? (": " + message) : ""))
    {
        gpuErrorType_ = gpuErrorType;
    }

	~query_engine_error()
	{
	}

    QueryEngineErrorType GetQueryEngineError()
    {
        return gpuErrorType_;
    }
};


/// <summary>
/// Check 'cudaError' and throw a cuda_error if it is not cudaSuccess
/// </summary>
void CheckCudaError(cudaError_t cudaError);

/// <summary>
/// Check 'errorType' and throw a query_engine_error if it is not GPU_EXTENSION_SUCCESS
/// </summary>
void CheckQueryEngineError(const QueryEngineErrorType errorType, const std::string& message = std::string());
