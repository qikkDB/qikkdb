#include "GPUError.h"
#include <iostream>
#include <cuda_runtime_api.h>

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

	if (cudaError != cudaSuccess)
	{
		std::cout << "CUDA Error " << cudaError << ": " << cudaGetErrorName(cudaError) << ", backtrace:" << std::endl;
#ifndef WIN32
		void* backtraceArray[25];
        int btSize = backtrace(backtraceArray, 25);
        char** symbols = backtrace_symbols(backtraceArray,btSize);
        for(int i = 0; i < btSize; i++)
        {
            std::cout << i << ": " << symbols[i] << std::endl;
        }
        std::cout << "---- backtrace end --------" << std::endl;
#endif
#ifdef DEBUG
		abort();
#endif
        throw cuda_error(cudaError);
    }
}

void CheckQueryEngineError(const QueryEngineErrorType errorType, const std::string& message)
{
    if (errorType != QueryEngineErrorType::GPU_EXTENSION_SUCCESS)
    {
        std::cout << "QueryEngineError " << errorType << ": " << message << ", backtrace:" << std::endl;
#ifndef WIN32
        void* backtraceArray[25];
        int btSize = backtrace(backtraceArray, 25);
        char** symbols = backtrace_symbols(backtraceArray,btSize);
        for(int i = 0; i < btSize; i++)
        {
            std::cout << i << ": " << symbols[i] << std::endl;
        }
        std::cout << "---- backtrace end --------" << std::endl;
#endif
        throw query_engine_error(errorType, message);
    }
}
