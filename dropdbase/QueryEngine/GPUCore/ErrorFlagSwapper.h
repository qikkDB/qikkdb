#pragma once

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
		GPUMemory::free(errorFlagPointer);
	}

	void Swap() {
		int32_t errorFlag;
		GPUMemory::copyDeviceToHost(&errorFlag, errorFlagPointer, 1);
		// Clear flag for repeatedly use of this object
		GPUMemory::memset(errorFlagPointer, static_cast<int32_t>(QueryEngineError::GPU_EXTENSION_SUCCESS), 1);

		if (errorFlag != QueryEngineError::GPU_EXTENSION_SUCCESS)
		{
			QueryEngineError::setType(static_cast<QueryEngineError::Type>(errorFlag));
		}
		else
		{
			QueryEngineError::setCudaError(cudaGetLastError());
		}
	}

	int32_t * GetFlagPointer() {
		return errorFlagPointer;
	}
};
