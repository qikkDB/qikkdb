#pragma once

#include "CompressionType.h"
#include <vector>
#include "GPUCompression.h"
#include "../DataType.h"

class Compression
{
public:
	
	template<class T>
	static void Compress(DataType columnType, T* const host_uncompressed, int64_t size, std::vector<T>& host_compressed, bool& compressedSuccessfully)
	{
		int64_t compressed_size;
		
		if (columnType == COLUMN_INT || columnType == COLUMN_LONG || columnType == COLUMN_INT8_T)
		{
			compressedSuccessfully = CompressionGPU::compressDataAAFL(host_uncompressed, size, host_compressed, compressed_size);
		}
		else
		{
			compressedSuccessfully = false;
		}
		
	}

	template<class T>
	static void Decompress(DataType columnType, T* const host_compressed, int64_t compressed_size, std::vector<T>& host_uncompressed, bool& compressedSuccessfully)
	{
		if (columnType == COLUMN_INT || columnType == COLUMN_LONG || columnType == COLUMN_INT8_T)
		{
			compressedSuccessfully = CompressionGPU::decompressDataAAFL(host_compressed, compressed_size, host_uncompressed);
		}
		else
		{
			compressedSuccessfully = false;
		}

	}

	template<class T>
	static size_t GetUncompressedDataSize(T* const host_compressed)
	{
		int64_t data_size = reinterpret_cast<int64_t*>(host_compressed)[0];
		return data_size;
	}

	template<class T>
	static size_t GetCompressedDataSize(T* const host_compressed)
	{
		int64_t compressed_data_size = reinterpret_cast<int64_t*>(host_compressed)[1];
		return compressed_data_size;
	}

	template<class T>
	static size_t GetCompressionBlocksCount(T* const host_compressed)
	{
		int64_t compression_blocks_count = reinterpret_cast<int64_t*>(host_compressed)[2];
		return compression_blocks_count;
	}

	template<class T>
	static void DecompressOnDevice(DataType columnType, T* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, T* const device_uncompressed, bool& compressedSuccessfully)
	{
		if (columnType == COLUMN_INT || columnType == COLUMN_LONG || columnType == COLUMN_INT8_T)
		{
			compressedSuccessfully = CompressionGPU::decompressDataAAFLOnDevice(device_compressed, data_size, compression_data_size, compression_blocks_count, device_uncompressed);
		}
		else
		{
			compressedSuccessfully = false;
		}

	}
};