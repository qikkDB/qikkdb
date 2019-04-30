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
	/// <param name="uncompressedElementsCount">Number of elements of compressed data</param>
	/// <param name="minValue">Minimum value of uncompressed data</param>
	/// <param name="maxValue">Maximum value of uncompressed data</param>
	/// <param name="compressedSuccessfully">Output parameter representing result of compression</param>
	template<class T>
	static void Decompress(DataType columnType, T* const hostCompressed, int64_t compressedElementsCount, std::vector<T>& hostUncompressed, int64_t uncompressedElementsCount, int64_t compressionBlocksCount, T minValue, T maxValue, bool& decompressedSuccessfully)
	{
		if (columnType == COLUMN_INT || columnType == COLUMN_LONG || columnType == COLUMN_INT8_T)
		{
			decompressedSuccessfully = CompressionGPU::decompressDataAAFL(hostCompressed, compressedElementsCount, hostUncompressed, uncompressedElementsCount, minValue, maxValue);			
		}
		else if (columnType == COLUMN_FLOAT)
		{
			decompressedSuccessfully = CompressionGPU::decompressDataAAFL(hostCompressed, compressedElementsCount, hostUncompressed, uncompressedElementsCount, minValue, maxValue);
		}
		else
		{
			decompressedSuccessfully = false;
		}
	}
	
	/// <summary>
	/// Decompresses input data and fills reserved space with decompressed data
	/// </summary>
	/// <param name="columnType">Type of column specified in DataType.h</param>
	/// <param name="hostCompressed">Pointer to compressed</param>
	/// <param name="compressedElementsCount">Number of elements of compressed data</param>
	/// <param name="hostUncompressed">Pointer to uncompressed data</param>
	/// <param name="compressedElementsCount">Number of elements of uncompressed data</param>
	/// <param name="compressedElementsCount">Number of compression blocks</param>
	/// <param name="minValue">Minimum value of uncompressed data</param>
	/// <param name="maxValue">Maximum value of uncompressed data</param>
	/// <param name="compressedSuccessfully">Output parameter representing result of compression</param>
	/// <param name="onDevice">Whether the decompression is on device (or on host)</param>
	template<class T>
	static void Decompress(DataType columnType, T* const deviceCompressed, int64_t compressedElementsCount, T* deviceUncompressed, int64_t uncompressedElementsCount, int64_t compressionBlocksCount, T minValue, T maxValue, bool& decompressedSuccessfully, bool onDevice = true)
	{
		if (!onDevice)
		{
			decompressedSuccessfully = false;
			return;
		}

		if (columnType == COLUMN_INT || columnType == COLUMN_LONG || columnType == COLUMN_INT8_T)
		{
			decompressedSuccessfully = CompressionGPU::decompressDataAAFLOnDevice(deviceCompressed, uncompressedElementsCount, compressedElementsCount, compressionBlocksCount, deviceUncompressed, minValue, maxValue);			
		}
		else if (columnType == COLUMN_FLOAT)
		{
			decompressedSuccessfully = CompressionGPU::decompressDataAAFLOnDevice(deviceCompressed, uncompressedElementsCount, compressedElementsCount, compressionBlocksCount, deviceUncompressed, minValue, maxValue);
		}
		else
		{
			decompressedSuccessfully = false;
		}
	}

	/// <summary>
	/// Extracts uncompressed elements count from compressed data
	/// </summary>
	/// <param name="hostCompresed">Pointer to compressed data stored in host memory</param>
	/// <returns>Uncompressed elements count</returns>
	template<class T>
	static size_t GetUncompressedDataElementsCount(T* const hostCompresed)
	{
		int64_t data_size = reinterpret_cast<int64_t*>(hostCompresed)[0];
		return data_size;
	}

	/// <summary>
	/// Extracts compressed elements count from compressed data
	/// </summary>
	/// <param name="hostCompresed">Pointer to compressed data stored in host memory</param>
	/// <returns>Compressed elements count</returns>
	template<class T>
	static size_t GetCompressedDataElementsCount(T* const hostCompresed)
	{
		int64_t compressed_data_size = reinterpret_cast<int64_t*>(hostCompresed)[1];
		return compressed_data_size;
	}

	/// <summary>
	/// Extracts compression blocks count from compressed data
	/// </summary>
	/// <param name="hostCompresed">Pointer to compressed data stored in host memory</param>
	/// <returns>Compression blocks count</returns>
	template<class T>
	static size_t GetCompressionBlocksCount(T* const hostCompresed)
	{
		int64_t compression_blocks_count = reinterpret_cast<int64_t*>(hostCompresed)[2];
		return compression_blocks_count;
	}
};