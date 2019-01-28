#ifndef ERROR_FLAG_SWAPPER_H
#define ERROR_FLAG_SWAPPER_H

#include <cstdint>

#include "../Context.h"
#include "GPUMemory.cuh"

class ErrorFlagSwapper {
private:
	int32_t *errorFlagPointer;

public:
	ErrorFlagSwapper() {
		GPUMemory::allocAndSet(&errorFlagPointer, static_cast<int32_t>(QueryEngineError::GPU_EXTENSION_SUCCESS), 1);
	}

	~ErrorFlagSwapper() {
		int32_t errorFlag;
		GPUMemory::copyDeviceToHost(&errorFlag, errorFlagPointer, 1);
		GPUMemory::free(errorFlagPointer);

		if (errorFlag != QueryEngineError::GPU_EXTENSION_SUCCESS)
		{
			Context::getInstance().getLastError().setType((QueryEngineError::Type)errorFlag);
		}
		else
		{
			Context::getInstance().getLastError().setCudaError(cudaGetLastError());
		}
	}

	int32_t * getFlagPointer() {
		return errorFlagPointer;
	}
};

#endif
