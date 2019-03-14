#pragma once

#include <cstdio>
#include <chrono>
#include <stdexcept>
#include <stdint.h>
#include <inttypes.h>
#include "../BlockBase.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "feathergpu/fl/containers.cuh"
#include "dropdbase/Compression/feathergpu/fl/default.cuh"

//#include "feathergpu/fl/aafl_compact.cuh"



class CompressionGPU
{


public:
    static const char CWARP_SIZE = 32;

    template<typename T>
	bool compressDataAAFL(T* const host_uncompressed, int64_t size, std::vector<T>& host_compressed, int64_t& compressed_size);

	
};
