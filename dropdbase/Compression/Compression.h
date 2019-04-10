#pragma once

#include "CompressionType.h"
#include <vector>
#include "GPUCompression.h"
#include "../DataType.h"

class Compression
{
public:
	
	/// <summary>
	/// Compresses input data and fills output vector with compressed data
	/// </summary>
	/// <param name="columnType">Type of column specified in DataType.h</param>
	/// <param name="hostUncompressed">Poiter to uncompressed data stored in host memory</param>
	/// <param name="uncompressedElementsCount">Number of elements of uncompressed data</param>
	/// <param name="hostCompressed">Compressed data vector in host memory</param>
	/// <param name="minValue">Minimum value of uncompressed data</param>
	/// <param name="maxValue">Maximum value of uncompressed data</param>
	/// <param name="compressedSuccessfully">Output parameter representing result of compression</param>
	template<class T>
	static void Compress(DataType columnType, T* const hostUncompressed, int64_t uncompressedElementsCount, std::vector<T>& hostCompressed, T minValue, T maxValue, bool& compressedSuccessfully)
	{
		int64_t compressedElementsCount;
		
		if (columnType == COLUMN_INT || columnType == COLUMN_LONG || columnType == COLUMN_INT8_T)
		{
			compressedSuccessfully = CompressionGPU::compressDataAAFL(hostUncompressed, uncompressedElementsCount, hostCompressed, compressedElementsCount, minValue, maxValue);
		}
		else if (columnType == COLUMN_FLOAT)
		{
			compressedSuccessfully = CompressionGPU::compressDataAAFL(hostUncompressed, uncompressedElementsCount, hostCompressed, compressedElementsCount, minValue, maxValue);
		}
		else
		{
			compressedSuccessfully = false;
		}
		
	}

	/// <summary>
	/// Decompresses input data and fills output vector with decompressed data
	/// </summary>
	/// <param name="columnType">Type of column specified in DataType.h</param>
	/// <param name="hostCompressed">Pointer to compressed data stored in host memory</param>
	/// <param name="compressedElementsCount">Number of elements of compressed data</param>
	/// <param name="hostUncompressed">Uncompressed data vector in host memory</param>
	/// <param name="minValue">Minimum value of uncompressed data</param>
	/// <param name="maxValue">Maximum value of uncompressed data</param>
	/// <param name="compressedSuccessfully">Output parameter representing result of compression</param>
	template<class T>
	static void Decompress(DataType columnType, T* const hostCompressed, int64_t compressedElementsCount, std::vector<T>& hostUncompressed, T minValue, T maxValue, bool& compressedSuccessfully)
	{
		int64_t uncompressedElementsCount;

		if (columnType == COLUMN_INT || columnType == COLUMN_LONG || columnType == COLUMN_INT8_T)
		{
			compressedSuccessfully = CompressionGPU::decompressDataAAFL(hostCompressed, compressedElementsCount, hostUncompressed, uncompressedElementsCount, minValue, maxValue);
		}
		else if (columnType == COLUMN_FLOAT)
		{
			compressedSuccessfully = CompressionGPU::decompressDataAAFL(hostCompressed, compressedElementsCount, hostUncompressed, uncompressedElementsCount, minValue, maxValue);
		}
		else
		{
			compressedSuccessfully = false;
		}

	}

	/// <summary>
	/// Decompresses input data directly on device and fills reserved space on device with decompressed data
	/// </summary>
	/// <param name="columnType">Type of column specified in DataType.h</param>
	/// <param name="deviceCompressed">Pointer to compressed data stored in device memory</param>
	/// <param name="compressedElementsCount">Number of elements of compressed data</param>
	/// <param name="hostUncompressed">Uncompressed data vector in host memory</param>
	/// <param name="minValue">Minimum value of uncompressed data</param>
	/// <param name="maxValue">Maximum value of uncompressed data</param>
	/// <param name="compressedSuccessfully">Output parameter representing result of compression</param>
	template<class T>
	static void DecompressOnDevice(DataType columnType, T* const deviceCompressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, T* const device_uncompressed, T minValue, T maxValue, bool& compressedSuccessfully)
	{
		if (columnType == COLUMN_INT || columnType == COLUMN_LONG || columnType == COLUMN_INT8_T)
		{
			compressedSuccessfully = CompressionGPU::decompressDataAAFLOnDevice(deviceCompressed, data_size, compression_data_size, compression_blocks_count, device_uncompressed, minValue, maxValue);
		}
		else if (columnType == COLUMN_FLOAT)
		{
			compressedSuccessfully = CompressionGPU::decompressDataAAFLOnDevice(deviceCompressed, data_size, compression_data_size, compression_blocks_count, device_uncompressed, minValue, maxValue);
		}
		else
		{
			compressedSuccessfully = false;
		}

	}

	/// <summary>
	/// Extracts uncompressed elements count from compressed data
	/// </summary>
	/// <param name="hostCompresed">Pointer to compressed data stored in host memory</param>
	/// <returns>Uncompressed elements count</returns>
	template<class T>
	static size_t GetUncompressedDataElementsCount(T* const host_compressed)
	{
		int64_t data_size = reinterpret_cast<int64_t*>(host_compressed)[0];
		return data_size;
	}

	/// <summary>
	/// Extracts compressed elements count from compressed data
	/// </summary>
	/// <param name="hostCompresed">Pointer to compressed data stored in host memory</param>
	/// <returns>Compressed elements count</returns>
	template<class T>
	static size_t GetCompressedDataElementsCount(T* const host_compressed)
	{
		int64_t compressed_data_size = reinterpret_cast<int64_t*>(host_compressed)[1];
		return compressed_data_size;
	}

	/// <summary>
	/// Extracts compression blocks count from compressed data
	/// </summary>
	/// <param name="hostCompresed">Pointer to compressed data stored in host memory</param>
	/// <returns>Compression blocks count</returns>
	template<class T>
	static size_t GetCompressionBlocksCount(T* const host_compressed)
	{
		int64_t compression_blocks_count = reinterpret_cast<int64_t*>(host_compressed)[2];
		return compression_blocks_count;
	}
};