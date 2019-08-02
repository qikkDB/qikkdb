#pragma once

#include <driver_types.h>
#include <string>
#include <stdexcept>

/// Enum for different QueryEngine errors
/// < param name="GPU_EXTENSION_SUCCESS"> Return code for successful operations</param>
/// < param name="GPU_EXTENSION_ERROR"> Return code for all CUDA errors</param>
/// < param name="GPU_DIVISION_BY_ZERO_ERROR"> Return code for division by zero</param>
/// < param name="GPU_INTEGER_OVERFLOW_ERROR"> Return code for integer overflow</param>
/// < param name="GPU_HASH_TABLE_FULL"> Return code for exceeding hash table limit (e.g. at group by with too many buckets)</param>
/// < param name="GPU_UNKNOWN_AGG_FUN"> The used function in the group by command is unknown</param>
/// < param name="GPU_NOT_FOUND_ERROR"> Return code for no detected GPU</param>
/// < param name="GPU_MEMORY_MAPPING_NOT_SUPPORTED_ERROR"> Return code for no memory mapping</param>
/// < param name="GPU_DRIVER_NOT_FOUND_EXCEPTION"> Return code for not found nvidia driver</param>
enum QueryEngineErrorType
{
    GPU_EXTENSION_SUCCESS = 0,
    GPU_EXTENSION_ERROR,
    GPU_DIVISION_BY_ZERO_ERROR,
    GPU_INTEGER_OVERFLOW_ERROR,
    GPU_HASH_TABLE_FULL,
    GPU_UNKNOWN_AGG_FUN,
    GPU_NOT_FOUND_ERROR,
    GPU_MEMORY_MAPPING_NOT_SUPPORTED_ERROR,
    GPU_DRIVER_NOT_FOUND_EXCEPTION
};

/// Generic GPU Error
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


/// Error for CUDA "internal" errors
class cuda_error : public gpu_error
{
private:
    cudaError_t cudaError_;

public:
    /// Create cuda_error from cudaError_t status
    /// and contain the number and the name of the error in an error message.
    /// <param name="cudaError">return value from cudaGetLastError()</param>
    explicit cuda_error(cudaError_t cudaError);


    ~cuda_error()
    {
    }

    /// Return stored value from cudaGetLastError()
    /// <return>value from cudaGetLastError()</return>
    cudaError_t GetCudaError() const
    {
        return cudaError_;
    }
};


/// Error for our QueryEngine errors
class query_engine_error : public gpu_error
{
private:
    QueryEngineErrorType queryEngineErrorType_;

public:
    /// Create query_engine_error from QueryEngineErrorType status and some extra message
    /// and contain the number and the extra message in error message.
    /// <param name="queryEngineErrorType">error type - value from enum QueryEngineErrorType</param>
    /// <param name="message">extra message (can be empty string "")</param>
    explicit query_engine_error(QueryEngineErrorType queryEngineErrorType, const std::string& message);

    ~query_engine_error()
    {
    }

    /// Return stored value of error type
    /// <return>stored error type - value from enum QueryEngineErrorType</return>
    QueryEngineErrorType GetQueryEngineError() const
    {
        return queryEngineErrorType_;
    }
};


/// Check 'cudaError' and throw a cuda_error if it is not cudaSuccess
/// <param name="cudaError">return value from cudaGetLastError()</param>
void CheckCudaError(cudaError_t cudaError);

/// Check 'errorType' and throw a query_engine_error if it is not GPU_EXTENSION_SUCCESS
/// <param name="errorType">error type - value from enum QueryEngineErrorType</param>
/// <param name="message">extra message (optional parameter)</param>
void CheckQueryEngineError(const QueryEngineErrorType errorType, const std::string& message = std::string());
