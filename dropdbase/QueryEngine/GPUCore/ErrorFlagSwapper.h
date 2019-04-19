#pragma once

#include <cstdint>

#include "../Context.h"
#include "GPUMemory.cuh"

/// <summary>
/// This class is used for GPU errors caused in a kernel. The kernel will set a flag and after run
/// of the kernel the flag can be copied by function Swap of this class. Finally, the flag will be
/// checked by GPUError and an error will be thrown if value of the flag is not success.
/// </summary>
class ErrorFlagSwapper
{
private:
    int32_t* errorFlagPointer;

public:
    ErrorFlagSwapper()
    {
        GPUMemory::allocAndSet(&errorFlagPointer,
                               static_cast<int32_t>(QueryEngineErrorType::GPU_EXTENSION_SUCCESS), 1);
    }

    ~ErrorFlagSwapper()
    {
        GPUMemory::free(errorFlagPointer);
    }

    void Swap()
    {
        int32_t errorFlag;
        GPUMemory::copyDeviceToHost(&errorFlag, errorFlagPointer, 1);
        // Clear flag for repeatedly use of this object
        GPUMemory::memset(errorFlagPointer,
                          static_cast<int32_t>(QueryEngineErrorType::GPU_EXTENSION_SUCCESS), 1);

        if (errorFlag != QueryEngineErrorType::GPU_EXTENSION_SUCCESS)
        {
            CheckQueryEngineError(static_cast<QueryEngineErrorType>(errorFlag));
        }
        else
        {
            CheckCudaError(cudaGetLastError());
        }
    }

    int32_t* GetFlagPointer()
    {
        return errorFlagPointer;
    }
};
