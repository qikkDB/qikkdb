#pragma once

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class cuda_error : public std::runtime_error
{
public:
	explicit cuda_error(const std::string& what_arg) : runtime_error(what_arg)
	{
	}
};


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

	void setCudaError(cudaError_t cudaError) {
		if (cudaError != cudaSuccess)
		{
			throw cuda_error(std::string(cudaGetErrorString(cudaError)));
		}
	}

	void setType(Type type) {
		if (type != GPU_EXTENSION_SUCCESS)
		{
			throw cuda_error("GPU Error number " + std::to_string(type));
		}
	}


};

