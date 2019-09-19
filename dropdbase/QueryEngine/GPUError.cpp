#include "GPUError.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include "../CudaLogBoost.h"
#include "Context.h"
#ifndef WIN32
#include <execinfo.h>
#endif // !WIN32


cuda_error::cuda_error(cudaError_t cudaError)
: gpu_error("CUDA Error " + std::to_string(static_cast<int32_t>(cudaError)) + ": " +
            std::string(cudaGetErrorName(cudaError)))
{
    cudaError_ = cudaError;
}

query_engine_error::query_engine_error(QueryEngineErrorType queryEngineErrorType, const std::string& message)
: gpu_error("GPU Error " + std::to_string(queryEngineErrorType) + (message.size() > 0 ? (": " + message) : ""))
{
    queryEngineErrorType_ = queryEngineErrorType;
}

void CheckCudaError(cudaError_t cudaError)
{
#ifdef DEBUG
    cudaDeviceSynchronize();
#endif // DEBUG
#ifdef DEBUG_ALLOC
    cudaDeviceSynchronize();
    Context::getInstance().ValidateCanariesForCurrentDevice();
#endif
    if (cudaError != cudaSuccess)
    {
        CudaLogBoost::getInstance(CudaLogBoost::error)
            << "CUDA Error " << cudaError << ": " << cudaGetErrorName(cudaError) << '\n';
#ifndef WIN32
        CudaLogBoost::getInstance(CudaLogBoost::debug) << "Backtrace:" << '\n';
        void* backtraceArray[25];
        int btSize = backtrace(backtraceArray, 25);
        char** symbols = backtrace_symbols(backtraceArray, btSize);
        for (int i = 0; i < btSize; i++)
        {
            CudaLogBoost::getInstance(CudaLogBoost::debug) << i << ": " << symbols[i] << '\n';
        }
        CudaLogBoost::getInstance(CudaLogBoost::debug) << "---- backtrace end --------" << '\n';
#endif
#ifdef DEBUG
        abort();
#endif
        throw cuda_error(cudaError);
    }
}

void CheckQueryEngineError(const QueryEngineErrorType errorType, const std::string& message)
{
#ifdef DEBUG_ALLOC
    cudaDeviceSynchronize();
    Context::getInstance().ValidateCanariesForCurrentDevice();
#endif
    if (errorType != QueryEngineErrorType::GPU_EXTENSION_SUCCESS)
    {
        CudaLogBoost::getInstance(CudaLogBoost::error)
            << "QueryEngineError " << errorType << ": " << message << "Backtrace:" << '\n';
#ifndef WIN32
        CudaLogBoost::getInstance(CudaLogBoost::debug) << "Backtrace:" << '\n';
        void* backtraceArray[25];
        int btSize = backtrace(backtraceArray, 25);
        char** symbols = backtrace_symbols(backtraceArray, btSize);
        for (int i = 0; i < btSize; i++)
        {
            CudaLogBoost::getInstance(CudaLogBoost::debug) << i << ": " << symbols[i] << '\n';
        }
        CudaLogBoost::getInstance(CudaLogBoost::debug) << "---- backtrace end --------" << '\n';
#endif
        throw query_engine_error(errorType, message);
    }
}
