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
	static bool compressDataAAFL(T* const host_uncompressed, int64_t size, std::vector<T>& host_compressed, int64_t& compressed_size, T min, T max);
	template<typename T>
	static bool decompressDataAAFL(T* const host_compressed, int64_t compressed_size, std::vector<T>& host_uncompressed, T min, T max);
	template<typename T>
	static bool decompressDataAAFLOnDevice(T* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, T* const device_uncompressed, T min, T max);

};
