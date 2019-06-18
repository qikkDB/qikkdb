#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "dropdbase/Compression/feathergpu/fl/containers.cuh"
#include "dropdbase/Compression/feathergpu/fl/default.cuh"
#include "GPUCompression.h"
#include "dropdbase/QueryEngine/Context.h" 
#include <memory>
#include <string>
#include <limits>
#include "dropdbase/Types/ComplexPolygon.pb.h"
#include "dropdbase/Types/Point.pb.h"

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
	// Sets offset for data transformation (subtracting minimal value), it checkes if it is possible to transform within range of type T
	T offset = minValue;
	if (minValue < 0 && maxValue > 0)
	{
		if (std::numeric_limits<T>::max() - maxValue < -minValue)
			offset = 0;
	}

	int64_t uncompressedDataSize = uncompressedElementsCount * sizeof(T); // size in bytes
	int64_t compressionBlocksCount = (uncompressedDataSize + (sizeof(T) * CWARP_SIZE) - 1) / (sizeof(T) * CWARP_SIZE);

	// Device pointers to compression data and metadata
	T *deviceUncompressed;
	T *deviceCompressed;
	unsigned char *deviceBitLength;
	unsigned long *devicePositionId;
	unsigned long *deviceCompressedElementsCount;

	// Device allocations for compression
	auto& cudaAllocator = Context::getInstance().GetAllocatorForCurrentDevice();
	deviceUncompressed = reinterpret_cast<T*>(cudaAllocator.allocate(uncompressedDataSize));
	deviceCompressed = reinterpret_cast<T*>(cudaAllocator.allocate(uncompressedDataSize)); // first we do not know what will be the size, therfore data_size	
	deviceBitLength = reinterpret_cast<unsigned char*>(cudaAllocator.allocate(compressionBlocksCount * sizeof(unsigned char)));
	devicePositionId = reinterpret_cast<unsigned long*>(cudaAllocator.allocate(compressionBlocksCount * sizeof(unsigned long)));
	deviceCompressedElementsCount = reinterpret_cast<unsigned long*>(cudaAllocator.allocate(sizeof(unsigned long)));
		
	// Copy data CPU->GPU
	cudaMemcpy(deviceUncompressed, hostUncompressed, uncompressedDataSize, cudaMemcpyHostToDevice);
	CheckCudaError(cudaGetLastError());

	// Set before compression
	cudaMemset(deviceCompressed, 0, uncompressedDataSize);
	cudaMemset(deviceCompressedElementsCount, 0, sizeof(unsigned long));
	cudaMemset(deviceBitLength, 0, compressionBlocksCount * sizeof(unsigned char));
	cudaMemset(devicePositionId, 0, compressionBlocksCount * sizeof(unsigned long));
	CheckCudaError(cudaGetLastError());

	// Compression
	container_uncompressed<T> udata = { deviceUncompressed, static_cast<unsigned long>(uncompressedElementsCount) };
	container_aafl<T> cdata = { deviceCompressed, static_cast<unsigned long>(uncompressedElementsCount), deviceBitLength, devicePositionId, deviceCompressedElementsCount, offset };
	gpu_fl_naive_launcher_compression<T, 32, container_aafl<T>>::compress(udata, cdata);
	CheckCudaError(cudaGetLastError());

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

		int positionCodedDataPositionId = (sizeof(int64_t) * 3);
		int positionCodedDataBitLength = positionCodedDataPositionId + (sizeof(unsigned long) * compressionBlocksCount);
		int positionHostOut = positionCodedDataBitLength + (sizeof(char) * compressionBlocksCount);

		hostCompressed.reserve(compressedDataSizeTotal / sizeof(T));
		
		// Resulting pointer to host compressed data
		std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[(compressedDataSizeTotal / sizeof(T))]);

		// Copy all compression data GPU -> CPU
		std::move(reinterpret_cast<char*>(codedSizes), reinterpret_cast<char*>(codedSizes) + (sizeof(int64_t) * 3), reinterpret_cast<char*>(data.get()));
		cudaMemcpy(reinterpret_cast<char*>(data.get()) + positionCodedDataPositionId, devicePositionId, compressionBlocksCount * sizeof(unsigned long), cudaMemcpyDeviceToHost);
		cudaMemcpy(reinterpret_cast<char*>(data.get()) + positionCodedDataBitLength, deviceBitLength, compressionBlocksCount * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaMemcpy(reinterpret_cast<char*>(data.get()) + positionHostOut, deviceCompressed, compressedDataSize, cudaMemcpyDeviceToHost);
		CheckCudaError(cudaGetLastError());

		// Assignment into output parameter
		hostCompressed.assign(data.get(), data.get() + compressedElementsCount);

		result = true;
	}
	else
	{
		result = false;
	}

	// Clean up device allocations
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceUncompressed), uncompressedDataSize);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceCompressed), uncompressedDataSize);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceBitLength), compressionBlocksCount * sizeof(unsigned char));
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(devicePositionId), compressionBlocksCount * sizeof(unsigned long));
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceCompressedElementsCount), sizeof(long));
	CheckCudaError(cudaGetLastError());

	return result;
}


template<>
bool CompressionGPU::compressDataAAFL<int32_t>(int32_t* const hostUncompressed, int64_t uncompressedElementsCount, std::vector<int32_t>& hostCompressed, int64_t& compressedElementsCount, int32_t minValue, int32_t maxValue)
{
	return compressAAFL(32, hostUncompressed, uncompressedElementsCount, hostCompressed, compressedElementsCount, minValue, maxValue);
}

template<>
bool CompressionGPU::compressDataAAFL<int64_t>(int64_t* const hostUncompressed, int64_t uncompressedElementsCount, std::vector<int64_t>& hostCompressed, int64_t& compressedElementsCount, int64_t minValue, int64_t maxValue)
{
	return compressAAFL(32, hostUncompressed, uncompressedElementsCount, hostCompressed, compressedElementsCount, minValue, maxValue);
}

template<>
bool CompressionGPU::compressDataAAFL<int8_t>(int8_t* const hostUncompressed, int64_t uncompressedElementsCount, std::vector<int8_t>& hostCompressed, int64_t& compressedElementsCount, int8_t minValue, int8_t maxValue)
{
	return compressAAFL(32, hostUncompressed, uncompressedElementsCount, hostCompressed, compressedElementsCount, minValue, maxValue);
}

template<>
bool CompressionGPU::compressDataAAFL<double>(double* const hostUncompressed, int64_t uncompressedElementsCount, std::vector<double>& hostCompressed, int64_t& compressedElementsCount, double minValue, double maxValue)
{	
	return 0;
}

template<>
bool CompressionGPU::compressDataAAFL<float>(float* const hostUncompressed, int64_t uncompressedElementsCount, std::vector<float>& hostCompressed, int64_t& compressedElementsCount, float minValue, float maxValue)
{
	if ((minValue >= 0 && maxValue >= 0) || (minValue <= 0 && maxValue <= 0))
	{
		int32_t * host_uncompressed_int32 = reinterpret_cast<int32_t*>(hostUncompressed);
		std::vector<int32_t> host_compressed_int32;
		int32_t min_int32 = *(reinterpret_cast<int32_t*>(&minValue));
		int32_t max_int32 = *(reinterpret_cast<int32_t*>(&maxValue));
		if (min_int32 > max_int32)
			std::swap(min_int32, max_int32);
		bool compressed = compressAAFL(32, host_uncompressed_int32, uncompressedElementsCount, host_compressed_int32, compressedElementsCount, min_int32, max_int32);
		if (compressed) {
			const int32_t *p_host_compressed_int32 = host_compressed_int32.data();
			hostCompressed.reserve(compressedElementsCount);
			const float *p_host_compressed = reinterpret_cast<const float *>(p_host_compressed_int32);
			hostCompressed.assign(p_host_compressed, p_host_compressed + compressedElementsCount);
			return true;
		}
	}
	
	return 0;
}

template<>
bool CompressionGPU::compressDataAAFL<std::string>(std::string* const hostUncompressed, int64_t uncompressedElementsCount, std::vector<std::string>& hostCompressed, int64_t& compressedElementsCount, std::string minValue, std::string maxValue)
{
	return 0;
}


template<>
bool CompressionGPU::compressDataAAFL<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon* const hostUncompressed, int64_t uncompressedElementsCount, std::vector<ColmnarDB::Types::ComplexPolygon>& hostCompressed, int64_t& compressedElementsCount, ColmnarDB::Types::ComplexPolygon minValue, ColmnarDB::Types::ComplexPolygon maxValue)
{
	return 0;
}

template<>
bool CompressionGPU::compressDataAAFL<ColmnarDB::Types::Point>(ColmnarDB::Types::Point* const hostUncompressed, int64_t uncompressedElementsCount, std::vector<ColmnarDB::Types::Point>& hostCompressed, int64_t& compressedElementsCount, ColmnarDB::Types::Point minValue, ColmnarDB::Types::Point maxValue)
{
	return 0;
}




/// <summary>
/// Decompresses input data and fills output vector with decompressed data
/// </summary>
/// <param name="CWARP_SIZE">Warp size</param>
/// <param name="hostCompressed">Pointer to compressed data stored in host memory</param>
/// <param name="compressedElementsCount">Number of elements of compressed data</param>
/// <param name="hostUncompressed">Uncompressed data vector in host memory</param>
/// <param name="uncompressedElementsCount">Number of elements of uncompressed data</param>
/// <param name="minValue">Minimum value of uncompressed data</param>
/// <param name="maxValue">Maximum value of uncompressed data</param>
/// <returns>Value representing result of decompression</returns>
template<typename T>
bool decompressAAFL(const int CWARP_SIZE, T* const hostCompressed, int64_t compressedElementsCount, std::vector<T>& hostUncompressed, int64_t &uncompressedElementsCount, T minValue, T maxValue)
{
	// Sets offset for data transformation (subtracting minimal value), it checkes if it is possible to transform within range of type T
	T offset = minValue;
	if (minValue < 0 && maxValue > 0)
	{
		if (std::numeric_limits<T>::max() - maxValue < -minValue)
			offset = 0;
	}

	uncompressedElementsCount = reinterpret_cast<int64_t*>(hostCompressed)[0];
	compressedElementsCount = reinterpret_cast<int64_t*>(hostCompressed)[1];
	int64_t compressionBlocksCount = reinterpret_cast<int64_t*>(hostCompressed)[2];

	int64_t uncompressedDataSize = uncompressedElementsCount * sizeof(T); // size in bytes
	int64_t compressedDataSize = compressedElementsCount * sizeof(T); // size in bytes

	// Device pointers to compression data and metadata
	T *deviceUncompressed;
	T *deviceCompressed;
	unsigned char *deviceBitLength;
	unsigned long *devicePositionId;

	// Device allocations for decompression
	auto& cudaAllocator = Context::getInstance().GetAllocatorForCurrentDevice();
	deviceUncompressed = reinterpret_cast<T*>(cudaAllocator.allocate(uncompressedDataSize));
	deviceCompressed = reinterpret_cast<T*>(cudaAllocator.allocate(compressedDataSize));
	deviceBitLength = reinterpret_cast<unsigned char*>(cudaAllocator.allocate(compressionBlocksCount * sizeof(unsigned char)));
	devicePositionId = reinterpret_cast<unsigned long*>(cudaAllocator.allocate(compressionBlocksCount * sizeof(unsigned long)));
	CheckCudaError(cudaGetLastError());

	// Decoding single array of type T into separate arrays (of compression meta data)
	int positionCodedDataPositionId = 3*sizeof(int64_t);
	int positionCodedDataBitLength = positionCodedDataPositionId + (sizeof(unsigned long) * compressionBlocksCount);
	int positionHostOut = positionCodedDataBitLength + (compressionBlocksCount);

	unsigned long *hostPositionId = reinterpret_cast<unsigned long*>(reinterpret_cast<char*>(hostCompressed) + positionCodedDataPositionId);
	unsigned char *hostBitLength = reinterpret_cast<unsigned char*>(hostCompressed) + positionCodedDataBitLength;
	T *hostCompressedValuesData = reinterpret_cast<T*>(reinterpret_cast<char*>(hostCompressed) + positionHostOut) ; // data of values only without meta data

	// Copy data CPU->GPU
	cudaMemcpy(deviceCompressed, hostCompressedValuesData, compressedDataSize - (positionHostOut), cudaMemcpyHostToDevice); // from compression size we need to subtract leading bytes with meta info
	cudaMemcpy(devicePositionId, hostPositionId, compressionBlocksCount * sizeof(unsigned long), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceBitLength, hostBitLength, compressionBlocksCount * sizeof(unsigned char), cudaMemcpyHostToDevice);
	CheckCudaError(cudaGetLastError());

	// Decompression
	container_uncompressed<T> udata = { deviceUncompressed, static_cast<unsigned long>(uncompressedElementsCount) };
	container_aafl<T> cdata = { deviceCompressed, static_cast<unsigned long>(uncompressedElementsCount), deviceBitLength, devicePositionId, NULL, offset };
	gpu_fl_naive_launcher_decompression<T, 32, container_aafl<T>>::decompress(cdata, udata);
	CheckCudaError(cudaGetLastError());
	
	// Copy result GPU->CPU into resulting pointer
	std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[uncompressedDataSize / sizeof(T)]);
	cudaMemcpy(data.get(), deviceUncompressed, uncompressedDataSize, cudaMemcpyDeviceToHost);
	CheckCudaError(cudaGetLastError());

	// Clean up device allocations
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceUncompressed), uncompressedDataSize);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceCompressed), uncompressedDataSize);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(deviceBitLength), compressionBlocksCount * sizeof(unsigned char));
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(devicePositionId), compressionBlocksCount * sizeof(unsigned long));
	CheckCudaError(cudaGetLastError());

	// Assignment into output parameter
	hostUncompressed.reserve(uncompressedElementsCount);
	hostUncompressed.assign(data.get(), data.get() + uncompressedElementsCount);

	return true;
}

template<>
bool CompressionGPU::decompressDataAAFL<int32_t>(int32_t* const hostCompressed, int64_t compressedElementsCount, std::vector<int32_t>& hostUncompressed, int64_t &uncompressedElementsCount, int32_t minValue, int32_t maxValue)
{
	return decompressAAFL(32, hostCompressed, compressedElementsCount, hostUncompressed, uncompressedElementsCount, minValue, maxValue);	
}

template<>
bool CompressionGPU::decompressDataAAFL<int64_t>(int64_t* const hostCompressed, int64_t compressedElementsCount, std::vector<int64_t>& hostUncompressed, int64_t &uncompressedElementsCount, int64_t minValue, int64_t maxValue)
{
	return decompressAAFL(32, hostCompressed, compressedElementsCount, hostUncompressed, uncompressedElementsCount, minValue, maxValue);
}

template<>
bool CompressionGPU::decompressDataAAFL<int8_t>(int8_t* const hostCompressed, int64_t compressedElementsCount, std::vector<int8_t>& hostUncompressed, int64_t &uncompressedElementsCount, int8_t minValue, int8_t maxValue)
{
	return decompressAAFL(32, hostCompressed, compressedElementsCount, hostUncompressed, uncompressedElementsCount, minValue, maxValue);
}

template<>
bool CompressionGPU::decompressDataAAFL<float>(float* const hostCompressed, int64_t compressedElementsCount, std::vector<float>& hostUncompressed, int64_t &uncompressedElementsCount, float minValue, float maxValue)
{
	if ((minValue > 0 && maxValue > 0) || (minValue < 0 && maxValue < 0))
	{
		int32_t * host_compressed_int32 = reinterpret_cast<int32_t*>(hostCompressed);
		std::vector<int32_t> host_uncompressed_int32;
		int32_t min_int32 = *(reinterpret_cast<int32_t*>(&minValue));
		int32_t max_int32 = *(reinterpret_cast<int32_t*>(&maxValue));
		if (min_int32 > max_int32)
			std::swap(min_int32, max_int32);
		bool decompressed = decompressAAFL(32, host_compressed_int32, compressedElementsCount, host_uncompressed_int32, uncompressedElementsCount, min_int32, max_int32);
		if (decompressed) {
			const int32_t *p_host_uncompressed_int32 = host_uncompressed_int32.data();
			hostUncompressed.reserve(uncompressedElementsCount);
			const float *p_host_uncompressed = reinterpret_cast<const float *>(p_host_uncompressed_int32);
			hostUncompressed.assign(p_host_uncompressed, p_host_uncompressed + uncompressedElementsCount);
			return true;
		}
	}
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFL<double>(double* const hostCompressed, int64_t compressedElementsCount, std::vector<double>& hostUncompressed, int64_t &uncompressedElementsCount, double minValue, double maxValue)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFL<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon* const hostCompressed, int64_t compressedElementsCount, std::vector<ColmnarDB::Types::ComplexPolygon>& hostUncompressed, int64_t &uncompressedElementsCount, ColmnarDB::Types::ComplexPolygon minValue, ColmnarDB::Types::ComplexPolygon maxValue)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFL<ColmnarDB::Types::Point>(ColmnarDB::Types::Point* const hostCompressed, int64_t compressedElementsCount, std::vector<ColmnarDB::Types::Point>& hostUncompressed, int64_t &uncompressedElementsCount, ColmnarDB::Types::Point minValue, ColmnarDB::Types::Point maxValue)
{
	return false;
}





/// <summary>
	/// Decompresses input data directly on device and fills reserved space on device with decompressed data
	/// </summary>
	/// <param name="CWARP_SIZE">Warp size</param>	
	/// <param name="deviceCompressed">Pointer to compressed data stored in device memory</param>
	/// <param name="uncompressedElementsCount">Number of elements of uncompressed data</param>
	/// <param name="compressedElementsCount">Number of elements of compressed data</param>
	/// <param name="compressionBlocksCount">Number of elements of compression blocks</param>
	/// <param name="deviceUncompressed">Pointer to compressed data stored in device memory</param>
	/// <param name="minValue">Minimum value of uncompressed data</param>
	/// <param name="maxValue">Maximum value of uncompressed data</param>
	/// <param name="compressedSuccessfully">Output parameter representing result of decompression</param>
template<typename T>
bool decompressAAFLOnDevice(const int CWARP_SIZE, T* const deviceCompressed, int64_t uncompressedElementsCount, int64_t compressedElementsCount, int64_t compressionBlocksCount, T* const deviceUncompressed, T minValue, T maxValue)
{
	// Sets offset for data transformation (subtracting minimal value), it checkes if it is possible to transform within range of type T
	T offset = minValue;
	if (minValue < 0 && maxValue > 0)
	{
		if (std::numeric_limits<T>::max() - maxValue < -minValue)
			offset = 0;
	}

	// Decoding single array of type T into separate arrays (of compression meta data)
	int positionCodedDataPositionId = (sizeof(int64_t) * 3);
	int positionCodedDataBitLength = positionCodedDataPositionId + (sizeof(unsigned long) * compressionBlocksCount);
	int positionDeviceOut = positionCodedDataBitLength + compressionBlocksCount;

	unsigned long *devicePositionId = reinterpret_cast<unsigned long*>(reinterpret_cast<char*>(deviceCompressed) + positionCodedDataPositionId);
	unsigned char *deviceBitLength = reinterpret_cast<unsigned char*>(deviceCompressed) + positionCodedDataBitLength;
	T *deviceCompressedValuesData = reinterpret_cast<T*>(reinterpret_cast<char*>(deviceCompressed) + positionDeviceOut); // data of values only without meta data

	// Decompression
	container_uncompressed<T> udata = { deviceUncompressed, static_cast<unsigned long>(uncompressedElementsCount) };
	container_aafl<T> cdata = { deviceCompressedValuesData, static_cast<unsigned long>(uncompressedElementsCount), deviceBitLength, devicePositionId, NULL, offset };
	gpu_fl_naive_launcher_decompression<T, 32, container_aafl<T>>::decompress(cdata, udata);
	CheckCudaError(cudaGetLastError());

	return true;
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<int32_t>(int32_t* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, int32_t* const device_uncompressed, int32_t minValue, int32_t maxValue)
{
	return decompressAAFLOnDevice(32, device_compressed, data_size, compression_data_size, compression_blocks_count, device_uncompressed, minValue, maxValue);
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<int64_t>(int64_t* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, int64_t* const device_uncompressed, int64_t minValue, int64_t maxValue)
{
	return decompressAAFLOnDevice(32, device_compressed, data_size, compression_data_size, compression_blocks_count, device_uncompressed, minValue, maxValue);
}


template<>
bool CompressionGPU::decompressDataAAFLOnDevice<int8_t>(int8_t* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, int8_t* const device_uncompressed, int8_t minValue, int8_t maxValue)
{
	return decompressAAFLOnDevice(32, device_compressed, data_size, compression_data_size, compression_blocks_count, device_uncompressed, minValue, maxValue);
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<float>(float* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, float* const device_uncompressed, float minValue, float maxValue)
{
	if ((minValue > 0 && maxValue > 0) || (minValue < 0 && maxValue < 0))
	{
		int32_t * device_compressed_int32 = reinterpret_cast<int32_t*>(device_compressed);
		int32_t * device_uncompressed_int32 = reinterpret_cast<int32_t*>(device_uncompressed);
		int32_t min_int32 = *(reinterpret_cast<int32_t*>(&minValue));
		int32_t max_int32 = *(reinterpret_cast<int32_t*>(&maxValue));
		if (min_int32 > max_int32)
			std::swap(min_int32, max_int32);
		
		bool compressed = decompressAAFLOnDevice(32, device_compressed_int32, data_size, compression_data_size, compression_blocks_count, device_uncompressed_int32, min_int32, max_int32);
		return compressed;
	}
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<double>(double* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, double* const device_uncompressed, double minValue, double maxValue)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, ColmnarDB::Types::ComplexPolygon* const device_uncompressed, ColmnarDB::Types::ComplexPolygon minValue, ColmnarDB::Types::ComplexPolygon maxValue)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFLOnDevice<ColmnarDB::Types::Point>(ColmnarDB::Types::Point* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, ColmnarDB::Types::Point* const device_uncompressed, ColmnarDB::Types::Point minValue, ColmnarDB::Types::Point maxValue)
{
	return false;
}