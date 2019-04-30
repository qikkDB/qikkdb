#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "dropdbase/Compression/feathergpu/fl/containers.cuh"
#include "dropdbase/Compression/feathergpu/fl/default.cuh"
#include "GPUCompression.h"
#include "dropdbase/QueryEngine/Context.h" 
#include "dropdbase/QueryEngine/QueryEngineError.h" 
#include <memory>
#include <string>
#include <limits>
#include "dropdbase/Types/ComplexPolygon.pb.h"
#include "dropdbase/Types/Point.pb.h"
#include "dropdbase/QueryEngine/QueryEngineError.h"

/// <summary>
/// Compresses input data and fills output vector with compressed data
/// </summary>
/// <param name="CWARP_SIZE">Warp size</param>
/// <param name="hostUncompressed">Poiter to uncompressed data stored in host memory</param>
/// <param name="uncompressedElementsCount">Number of elements of uncompressed data</param>
/// <param name="hostCompressed">Compressed data vector in host memory</param>
/// <param name="compressedElementsCount">Number of elements of compressed data</param>
/// <param name="minValue">Minimum value of uncompressed data</param>
/// <param name="maxValue">Maximum value of uncompressed data</param>
/// <returns>Output parameter representing result of compression</returns>
template<typename T>
bool compressAAFL(const int CWARP_SIZE, T* const hostUncompressed, int64_t uncompressedElementsCount, std::vector<T>& hostCompressed, int64_t& compressedElementsCount, T minValue, T maxValue)
{
	// sets offset for data transformation (subtracting minimal value), it checkes if it is possible to transform within range of type T
	T offset = minValue;
	if (minValue < 0 && maxValue > 0)
	{
		if (std::numeric_limits<T>::max() - maxValue < -minValue)
			offset = 0;
	}

	int64_t uncompressedDataSize = uncompressedElementsCount * sizeof(T);
	int64_t compressionBlocksCount = (uncompressedDataSize + (sizeof(T) * CWARP_SIZE) - 1) / (sizeof(T) * CWARP_SIZE);


	T *deviceUncompressed;
	T *deviceCompressed;
	unsigned char *deviceBitLength;
	unsigned long *devicePositionId;
	unsigned long *deviceCompressedElementsCount;

	// device allocations for compression
	auto& cudaAllocator = Context::getInstance().GetAllocatorForCurrentDevice();
	deviceUncompressed = reinterpret_cast<T*>(cudaAllocator.allocate(uncompressedDataSize));
	deviceCompressed = reinterpret_cast<T*>(cudaAllocator.allocate(uncompressedDataSize)); // first we do not know what will be the size, therfore data_size	
	deviceBitLength = reinterpret_cast<unsigned char*>(cudaAllocator.allocate(compressionBlocksCount * sizeof(unsigned char)));
	devicePositionId = reinterpret_cast<unsigned long*>(cudaAllocator.allocate(compressionBlocksCount * sizeof(unsigned long)));
	deviceCompressedElementsCount = reinterpret_cast<unsigned long*>(cudaAllocator.allocate(sizeof(unsigned long)));
		
	//copy data CPU->GPU
	cudaMemcpy(deviceUncompressed, hostUncompressed, uncompressedDataSize, cudaMemcpyHostToDevice);
	QueryEngineError::setCudaError(cudaGetLastError());

	// Set before compression
	cudaMemset(deviceCompressed, 0, uncompressedDataSize);
	cudaMemset(deviceCompressedElementsCount, 0, sizeof(unsigned long));
	cudaMemset(deviceBitLength, 0, compressionBlocksCount * sizeof(unsigned char));
	cudaMemset(devicePositionId, 0, compressionBlocksCount * sizeof(unsigned long));
	QueryEngineError::setCudaError(cudaGetLastError());

	// Compression
	container_uncompressed<T> udata = { deviceUncompressed, uncompressedElementsCount };
	container_aafl<T> cdata = { deviceCompressed, uncompressedElementsCount, deviceBitLength, devicePositionId, deviceCompressedElementsCount, offset };
	gpu_fl_naive_launcher_compression<T, 32, container_aafl<T>>::compress(udata, cdata);
	QueryEngineError::setCudaError(cudaGetLastError());

	// Gets compression elements (values only without meta data) count into RAM
	unsigned long compressedValuesCount;
	cudaMemcpy(&compressedValuesCount, deviceCompressedElementsCount, sizeof(unsigned long), cudaMemcpyDeviceToHost);
	int64_t compressedDataSize = (compressedValuesCount) * sizeof(T);

	// Total compression data size (values and meta data)
	unsigned long compressedDataSizeTotal = 
		compressedDataSize + 
		std::max(sizeof(unsigned long) * compressionBlocksCount, sizeof(T)) +
		std::max(sizeof(unsigned char) * compressionBlocksCount, sizeof(T)) +
		(sizeof(int64_t) * 3);
	compressedElementsCount = compressedDataSizeTotal / sizeof(T);

	bool result = false;
	// Does compression make sense?
	if (compressedElementsCount < uncompressedElementsCount)
	{
		// All data are coded into single array of type T
		// We determine positions of partials arrays (meta data) in one T array
		int64_t sizes[3] = { uncompressedElementsCount , compressedElementsCount, compressionBlocksCount };
		T* codedSizes = reinterpret_cast<T*>(sizes);

		int positionCodedDataPositionId = (sizeof(int64_t) / (float)sizeof(T) * 3);
		int positionCodedDataBitLength = positionCodedDataPositionId + std::max((int)(sizeof(unsigned long) / (float)sizeof(T) * compressionBlocksCount), 1);
		int positionHostOut = positionCodedDataBitLength + std::max((int)(sizeof(char) / (float)sizeof(T) * compressionBlocksCount), 1);

		hostCompressed.reserve(compressedDataSizeTotal / sizeof(T));
		
		// Resulting pointer to host compressed data
		std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[(compressedDataSizeTotal / sizeof(T))]);

		// Copy all compression data GPU -> CPU
		std::move(codedSizes, codedSizes + (int)(sizeof(int64_t) / (float)sizeof(T) * 3), data.get());
		cudaMemcpy(data.get() + positionCodedDataPositionId, devicePositionId, compressionBlocksCount * sizeof(unsigned long), cudaMemcpyDeviceToHost);
		cudaMemcpy(data.get() + positionCodedDataBitLength, deviceBitLength, compressionBlocksCount * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaMemcpy(data.get() + positionHostOut, deviceCompressed, compressedDataSize, cudaMemcpyDeviceToHost);

		// Assignment into output parameter
		hostCompressed.assign(data.get(), data.get() + compressedElementsCount);

		result = true;
	}
	else
	{
		result = false;
	}

	// clean up device allocations
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceUncompressed), uncompressedDataSize);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceCompressed), uncompressedDataSize);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceBitLength), compressionBlocksCount * sizeof(unsigned char));
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(devicePositionId), compressionBlocksCount * sizeof(unsigned long));
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceCompressedElementsCount), sizeof(long));
	QueryEngineError::setCudaError(cudaGetLastError());

	return result;
}





template<>
bool CompressionGPU::compressDataAAFL<int32_t>(int32_t* const host_uncompressed, int64_t size, std::vector<int32_t>& host_compressed, int64_t& compressed_size, int32_t min, int32_t max)
{
	return compressAAFL(32, host_uncompressed, size, host_compressed, compressed_size, min, max);
}

template<>
bool CompressionGPU::compressDataAAFL<int64_t>(int64_t* const host_uncompressed, int64_t size, std::vector<int64_t>& host_compressed, int64_t& compressed_size, int64_t min, int64_t max)
{
	return compressAAFL(32, host_uncompressed, size, host_compressed, compressed_size, min, max);
}

template<>
bool CompressionGPU::compressDataAAFL<int8_t>(int8_t* const host_uncompressed, int64_t size, std::vector<int8_t>& host_compressed, int64_t& compressed_size, int8_t min, int8_t max)
{
	return compressAAFL(32, host_uncompressed, size, host_compressed, compressed_size, min, max);
}

template<>
bool CompressionGPU::compressDataAAFL<double>(double* const host_uncompressed, int64_t size, std::vector<double>& host_compressed, int64_t& compressed_size, double min, double max)
{	
	return 0;
}

template<>
bool CompressionGPU::compressDataAAFL<float>(float* const host_uncompressed, int64_t size, std::vector<float>& host_compressed, int64_t& compressed_size, float min, float max)
{
	if ((min >= 0 && max >= 0) || (min <= 0 && max <= 0))
	{
		int32_t * host_uncompressed_int32 = reinterpret_cast<int32_t*>(host_uncompressed);
		std::vector<int32_t> host_compressed_int32;
		int32_t min_int32 = *(reinterpret_cast<int32_t*>(&min));
		int32_t max_int32 = *(reinterpret_cast<int32_t*>(&max));
		if (min_int32 > max_int32)
			std::swap(min_int32, max_int32);
		bool compressed = compressAAFL(32, host_uncompressed_int32, size, host_compressed_int32, compressed_size, min_int32, max_int32);
		if (compressed) {
			const int32_t *p_host_compressed_int32 = host_compressed_int32.data();
			host_compressed.reserve(compressed_size);
			const float *p_host_compressed = reinterpret_cast<const float *>(p_host_compressed_int32);
			host_compressed.assign(p_host_compressed, p_host_compressed + compressed_size);
			return true;
		}
	}
	
	return 0;
}

template<>
bool CompressionGPU::compressDataAAFL<std::string>(std::string* const host_uncompressed, int64_t size, std::vector<std::string>& host_compressed, int64_t& compressed_size, std::string min, std::string max)
{
	return 0;
}


template<>
bool CompressionGPU::compressDataAAFL<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon* const host_uncompressed, int64_t size, std::vector<ColmnarDB::Types::ComplexPolygon>& host_compressed, int64_t& compressed_size, ColmnarDB::Types::ComplexPolygon min, ColmnarDB::Types::ComplexPolygon max)
{
	return 0;
}

template<>
bool CompressionGPU::compressDataAAFL<ColmnarDB::Types::Point>(ColmnarDB::Types::Point* const host_uncompressed, int64_t size, std::vector<ColmnarDB::Types::Point>& host_compressed, int64_t& compressed_size, ColmnarDB::Types::Point min, ColmnarDB::Types::Point max)
{
	return 0;
}





template<typename T>
bool decompressAAFL(const int CWARP_SIZE, T* const host_compressed, int64_t compressed_size, std::vector<T>& host_uncompressed, int64_t &size, T min, T max)
{
	T offset = min;

	size = reinterpret_cast<int64_t*>(host_compressed)[0];
	compressed_size = reinterpret_cast<int64_t*>(host_compressed)[1];
	int64_t compression_blocks_count = reinterpret_cast<int64_t*>(host_compressed)[2];

	int64_t data_size = size * sizeof(T);
	int64_t compressed_data_size = compressed_size * sizeof(T);
	

	unsigned char *host_bit_length;
	unsigned long *host_position_id;

	T *host_compressed_data;

	T *device_uncompressed;
	T *device_compressed;
	unsigned char *device_bit_length;
	unsigned long *device_position_id;

	//allocations
	auto& cudaAllocator = Context::getInstance().GetAllocatorForCurrentDevice();
	device_uncompressed = reinterpret_cast<T*>(cudaAllocator.allocate(data_size));
	device_compressed = reinterpret_cast<T*>(cudaAllocator.allocate(compressed_data_size));
	device_bit_length = reinterpret_cast<unsigned char*>(cudaAllocator.allocate(compression_blocks_count * sizeof(unsigned char)));
	device_position_id = reinterpret_cast<unsigned long*>(cudaAllocator.allocate(compression_blocks_count * sizeof(unsigned long)));
	QueryEngineError::setCudaError(cudaGetLastError());

	int coded_data_position_id_start = (sizeof(int64_t) / (float)sizeof(T) * 3);
	int coded_data_bit_length_start = coded_data_position_id_start + std::max((int)(sizeof(unsigned long) / (float)sizeof(T) * compression_blocks_count), 1);
	int host_out_start = coded_data_bit_length_start + std::max((int)(sizeof(char) / (float)sizeof(T) * compression_blocks_count), 1);

	host_position_id = reinterpret_cast<unsigned long*>(&host_compressed[coded_data_position_id_start]);
	host_bit_length = reinterpret_cast<unsigned char*>(&host_compressed[coded_data_bit_length_start]);
	host_compressed_data = &host_compressed[host_out_start];

	cudaMemcpy(device_compressed, host_compressed_data, compressed_data_size - (host_out_start * sizeof(T)), cudaMemcpyHostToDevice); // from compression size we need to subtract leading bytes with meta info
	QueryEngineError::setCudaError(cudaGetLastError());
	cudaMemcpy(device_position_id, host_position_id, compression_blocks_count * sizeof(unsigned long), cudaMemcpyHostToDevice);
	QueryEngineError::setCudaError(cudaGetLastError());
	cudaMemcpy(device_bit_length, host_bit_length, compression_blocks_count * sizeof(unsigned char), cudaMemcpyHostToDevice);
	QueryEngineError::setCudaError(cudaGetLastError());

	container_uncompressed<T> udata = { device_uncompressed, size };
	container_aafl<T> cdata = { device_compressed, size, device_bit_length, device_position_id, NULL, offset };
	gpu_fl_naive_launcher_decompression<T, 32, container_aafl<T>>::decompress(cdata, udata);
	QueryEngineError::setCudaError(cudaGetLastError());
	
	std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[data_size / sizeof(T)]);
	cudaMemcpy(data.get(), device_uncompressed, data_size, cudaMemcpyDeviceToHost);
	QueryEngineError::setCudaError(cudaGetLastError());

	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_uncompressed), data_size);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_compressed), data_size);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_bit_length), compression_blocks_count * sizeof(unsigned char));
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_position_id), compression_blocks_count * sizeof(unsigned long));
	QueryEngineError::setCudaError(cudaGetLastError());

	host_uncompressed.reserve(size);
	host_uncompressed.assign(data.get(), data.get() + size);

	return true;
}

template<>
bool CompressionGPU::decompressDataAAFL<int32_t>(int32_t* const host_compressed, int64_t compressed_size, std::vector<int32_t>& host_uncompressed, int64_t &size, int32_t min, int32_t max)
{
	return decompressAAFL(32, host_compressed, compressed_size, host_uncompressed, size, min, max);	
}

template<>
bool CompressionGPU::decompressDataAAFL<int64_t>(int64_t* const host_compressed, int64_t compressed_size, std::vector<int64_t>& host_uncompressed, int64_t &size, int64_t min, int64_t max)
{
	return decompressAAFL(32, host_compressed, compressed_size, host_uncompressed, size, min, max);
}

template<>
bool CompressionGPU::decompressDataAAFL<int8_t>(int8_t* const host_compressed, int64_t compressed_size, std::vector<int8_t>& host_uncompressed, int64_t &size, int8_t min, int8_t max)
{
	return decompressAAFL(32, host_compressed, compressed_size, host_uncompressed, size, min, max);
}

template<>
bool CompressionGPU::decompressDataAAFL<float>(float* const host_compressed, int64_t compressed_size, std::vector<float>& host_uncompressed, int64_t &size, float min, float max)
{
	if ((min > 0 && max > 0) || (min < 0 && max < 0))
	{
		int32_t * host_compressed_int32 = reinterpret_cast<int32_t*>(host_compressed);
		std::vector<int32_t> host_uncompressed_int32;
		int32_t min_int32 = *(reinterpret_cast<int32_t*>(&min));
		int32_t max_int32 = *(reinterpret_cast<int32_t*>(&max));
		if (min_int32 > max_int32)
			std::swap(min_int32, max_int32);
		bool decompressed = decompressAAFL(32, host_compressed_int32, compressed_size, host_uncompressed_int32, size, min_int32, max_int32);
		if (decompressed) {
			const int32_t *p_host_uncompressed_int32 = host_uncompressed_int32.data();
			host_uncompressed.reserve(size);
			const float *p_host_uncompressed = reinterpret_cast<const float *>(p_host_uncompressed_int32);
			host_uncompressed.assign(p_host_uncompressed, p_host_uncompressed + size);
			return true;
		}
	}
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFL<double>(double* const host_compressed, int64_t compressed_size, std::vector<double>& host_uncompressed, int64_t &size, double min, double max)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFL<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon* const host_compressed, int64_t compressed_size, std::vector<ColmnarDB::Types::ComplexPolygon>& host_uncompressed, int64_t &size, ColmnarDB::Types::ComplexPolygon min, ColmnarDB::Types::ComplexPolygon max)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFL<ColmnarDB::Types::Point>(ColmnarDB::Types::Point* const host_compressed, int64_t compressed_size, std::vector<ColmnarDB::Types::Point>& host_uncompressed, int64_t &size, ColmnarDB::Types::Point min, ColmnarDB::Types::Point max)
{
	return false;
}






template<typename T>
bool decompressAAFLOnDevice(const int CWARP_SIZE, T* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, T* const device_uncompressed, T min, T max)
{
	T offset = min;

	
	T *device_compressed_data;

	unsigned char *device_bit_length;
	unsigned long *device_position_id;
		
	int coded_data_position_id_start = (sizeof(int64_t) / (float)sizeof(T) * 3);
	int coded_data_bit_length_start = coded_data_position_id_start + std::max((int)(sizeof(unsigned long) / (float)sizeof(T) * compression_blocks_count), 1);
	int device_out_start = coded_data_bit_length_start + std::max((int)(sizeof(char) / (float)sizeof(T) * compression_blocks_count), 1);

	device_position_id = reinterpret_cast<unsigned long*>(&device_compressed[coded_data_position_id_start]);
	device_bit_length = reinterpret_cast<unsigned char*>(&device_compressed[coded_data_bit_length_start]);
	device_compressed_data = &device_compressed[device_out_start];

	
	container_uncompressed<T> udata = { device_uncompressed, data_size };
	container_aafl<T> cdata = { device_compressed_data, data_size, device_bit_length, device_position_id, NULL, offset };

	gpu_fl_naive_launcher_decompression<T, 32, container_aafl<T>>::decompress(cdata, udata);
	QueryEngineError::setCudaError(cudaGetLastError());

	return true;
}


template<>
bool CompressionGPU::decompressDataAAFLOnDevice<int32_t>(int32_t* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, int32_t* const device_uncompressed, int32_t min, int32_t max)
{
	return decompressAAFLOnDevice(32, device_compressed, data_size, compression_data_size, compression_blocks_count, device_uncompressed, min, max);
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<int64_t>(int64_t* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, int64_t* const device_uncompressed, int64_t min, int64_t max)
{
	return decompressAAFLOnDevice(32, device_compressed, data_size, compression_data_size, compression_blocks_count, device_uncompressed, min, max);
}


template<>
bool CompressionGPU::decompressDataAAFLOnDevice<int8_t>(int8_t* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, int8_t* const device_uncompressed, int8_t min, int8_t max)
{
	return decompressAAFLOnDevice(32, device_compressed, data_size, compression_data_size, compression_blocks_count, device_uncompressed, min, max);
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<float>(float* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, float* const device_uncompressed, float min, float max)
{
	if ((min > 0 && max > 0) || (min < 0 && max < 0))
	{
		int32_t * device_compressed_int32 = reinterpret_cast<int32_t*>(device_compressed);
		int32_t * device_uncompressed_int32 = reinterpret_cast<int32_t*>(device_uncompressed);
		int32_t min_int32 = *(reinterpret_cast<int32_t*>(&min));
		int32_t max_int32 = *(reinterpret_cast<int32_t*>(&max));
		if (min_int32 > max_int32)
			std::swap(min_int32, max_int32);
		
		bool compressed = decompressAAFLOnDevice(32, device_compressed_int32, data_size, compression_data_size, compression_blocks_count, device_uncompressed_int32, min_int32, max_int32);
		return compressed;
	}
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<double>(double* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, double* const device_uncompressed, double min, double max)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, ColmnarDB::Types::ComplexPolygon* const device_uncompressed, ColmnarDB::Types::ComplexPolygon min, ColmnarDB::Types::ComplexPolygon max)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<ColmnarDB::Types::Point>(ColmnarDB::Types::Point* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, ColmnarDB::Types::Point* const device_uncompressed, ColmnarDB::Types::Point min, ColmnarDB::Types::Point max)
{
	return false;
}