#include "GPUError.h"


void CheckCudaError(cudaError_t cudaError)
{
    if (cudaError != cudaSuccess)
    {
        std::cout << "CUDA Error " << cudaError << ": " << cudaGetErrorName(cudaError) << std::endl;
        throw cuda_error(cudaError);
    }
}

void CheckQueryEngineError(const QueryEngineErrorType errorType, const std::string& message)
{
    if (errorType != QueryEngineErrorType::GPU_EXTENSION_SUCCESS)
    {
        std::cout << "QueryEngineError " << errorType << ": " << message << std::endl;
        throw query_engine_error(errorType, message);
    }
}
