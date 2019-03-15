#pragma once

#include <cstdio>
#include <chrono>
#include <stdexcept>
#include <stdint.h>
#include <inttypes.h>
#include <vector>
#include <cstdint>

class CompressionGPU
{
	
public:
    static const char CWARP_SIZE = 32;

    template<typename T>
	static bool compressDataAAFL(T* const host_uncompressed, int64_t size, std::vector<T>& host_compressed, int64_t& compressed_size);					
	template<typename T>
	static bool decompressDataAAFL(T* const host_compressed, int64_t compressed_size, std::vector<T>& host_uncompressed);
		
};
