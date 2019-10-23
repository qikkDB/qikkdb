#include "GPUError.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include "../CudaLogBoost.h"
#ifndef WIN32
#include <execinfo.h>
#endif // !WIN32


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
#endif // WIN32
        abort();
    }
}

void CheckQueryEngineError(const QueryEngineErrorType errorType, const std::string& message)
{
    if (errorType != QueryEngineErrorType::GPU_EXTENSION_SUCCESS)
    {
        CudaLogBoost::getInstance(CudaLogBoost::debug)
            << "QueryEngineError " << errorType << ": " << message << '\n';
#ifdef DEBUG
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
#endif // WIN32
#endif // DEBUG
        throw query_engine_error(errorType, message);
    }
}
