#ifndef ERROR_H
#define ERROR_H

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class QueryEngineError {
public:
	enum Type {
		GPU_EXTENSION_SUCCESS = 0,				// Return code for successful operations
		GPU_EXTENSION_ERROR,					// Return code for all CUDA errors
		GPU_DIVISION_BY_ZERO_ERROR,				// Return code for division by zero
		GPU_INTEGER_OVERFLOW_ERROR,				// Return code for integer overflow
		GPU_UNSUPPORTED_DATA_TYPE,				// Return code for unsupported data type
		GPU_BUFFER_TOO_BIG,						// Return code for too big buffer (e.g. at group by with too many buckets)
		GPU_NOT_FOUND_ERROR,					// Return code for no detected GPU
		GPU_MEMORY_MAPPING_NOT_SUPPORTED_ERROR,	// Return code for no memory mapping
		GPU_DRIVER_NOT_FOUND_EXCEPTION			// Return code for not found nvidia driver
	};

private:
	Type type_;
	std::string text_;

public:
	void setCudaError(cudaError_t cudaError) {
		switch (cudaError)
		{
		case cudaSuccess:
			// Don't overwrite last error with success
			break;
		default:
			type_ = GPU_EXTENSION_ERROR;
			text_ = cudaGetErrorString(cudaError);
			break;
		}
	}

	void setType(Type type) {
		if (type != GPU_EXTENSION_SUCCESS)
		{
			type_ = type;
		}
	}

	const Type getType() {
		Type last = type_;
		type_ = GPU_EXTENSION_SUCCESS;
		return last;
	}

	const std::string getText() {
		std::string last = text_;
		text_ = "";
		return last;
	}

};

#endif
