#pragma once

#include <string>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

class QueryEngineError {
public:
	enum Type {
		GPU_EXTENSION_SUCCESS = 0,				// Return code for successful operations
		GPU_EXTENSION_ERROR,					// Return code for all CUDA errors
		GPU_DIVISION_BY_ZERO_ERROR,				// Return code for division by zero
		GPU_INTEGER_OVERFLOW_ERROR,				// Return code for integer overflow
		GPU_HASH_TABLE_FULL,					// Return code for exceeding hash table limit (e.g. at group by with too many buckets)
		GPU_UNKNOWN_AGG_FUN,					// The used function in the group by command is unknown
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
			std::cout << cudaError << " " << cudaGetErrorString(cudaError) << '\n';
			throw cudaError;
			type_ = GPU_EXTENSION_ERROR;
			text_ = cudaGetErrorString(cudaError);
			std::cout << cudaError << " " << cudaGetErrorName(cudaError);
			throw;
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

