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
#include "dropdbase/QueryEngine/QueryEngineError.h"


template<typename T>
bool compressAAFL(const int CWARP_SIZE, T* const host_uncompressed, int64_t size, std::vector<T>& host_compressed, int64_t& compressed_size, T min, T max)
{
	T offset = min;
	if (min < 0)
	{
		if (std::numeric_limits<T>::max() - max < -min)
			offset = 0;
	}

	int64_t data_size = size * sizeof(T);
	int64_t compression_blocks_count = (data_size + (sizeof(T) * CWARP_SIZE) - 1) / (sizeof(T) * CWARP_SIZE);


	T *device_uncompressed;
	T *device_compressed;
	unsigned char *device_bit_length;
	unsigned long *device_position_id;
	unsigned long *device_compressed_size;

	//allocations
	auto& cudaAllocator = Context::getInstance().GetAllocatorForCurrentDevice();
	device_uncompressed = reinterpret_cast<T*>(cudaAllocator.allocate(data_size));
	device_compressed = reinterpret_cast<T*>(cudaAllocator.allocate(data_size)); // first we do not know what will be the size, therfore data_size
	
	device_bit_length = reinterpret_cast<unsigned char*>(cudaAllocator.allocate(compression_blocks_count * sizeof(unsigned char)));
	device_position_id = reinterpret_cast<unsigned long*>(cudaAllocator.allocate(compression_blocks_count * sizeof(unsigned long)));
	device_compressed_size = reinterpret_cast<unsigned long*>(cudaAllocator.allocate(sizeof(unsigned long)));
		
	//copy M->G
	cudaMemcpy(device_uncompressed, host_uncompressed, data_size, cudaMemcpyHostToDevice);

	// Clean up before compression
	cudaMemset(device_compressed, 0, data_size);
	cudaMemset(device_compressed_size, 0, sizeof(unsigned long));
	cudaMemset(device_bit_length, 0, compression_blocks_count * sizeof(unsigned char));
	cudaMemset(device_position_id, 0, compression_blocks_count * sizeof(unsigned long));

	container_uncompressed<T> udata = { device_uncompressed, size };
	container_aafl<T> cdata = { device_compressed, size, device_bit_length, device_position_id, device_compressed_size, offset };
	gpu_fl_naive_launcher_compression<T, 32, container_aafl<T>>::compress(udata, cdata);

	unsigned long host_compressed_size;
	cudaMemcpy(&host_compressed_size, device_compressed_size, sizeof(unsigned long), cudaMemcpyDeviceToHost);
	int64_t compressed_data_size = (host_compressed_size) * sizeof(T);

	// coding into single array
	unsigned long compressed_data_size_final = 
		compressed_data_size + 
		std::max(sizeof(unsigned long) * compression_blocks_count, sizeof(T)) +
		std::max(sizeof(unsigned char) * compression_blocks_count, sizeof(T)) +
		(sizeof(int64_t) * 3);
	compressed_size = compressed_data_size_final / sizeof(T);

	bool result = false;
	if (compressed_size < size)
	{
		int64_t sizes[3] = { data_size , compressed_data_size, compression_blocks_count };

		T* coded_sizes = reinterpret_cast<T*>(sizes);

		int coded_data_position_id_start = (sizeof(int64_t) / sizeof(T) * 3);
		int coded_data_bit_length_start = coded_data_position_id_start + std::max((int)(sizeof(unsigned long) / (float)sizeof(T) * compression_blocks_count), 1);
		int host_out_start = coded_data_bit_length_start + std::max((int)(sizeof(char) / (float)sizeof(T) * compression_blocks_count), 1);

		host_compressed.reserve(compressed_data_size_final / sizeof(T));
		
		std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[(compressed_data_size_final / sizeof(T))]);

		std::move(coded_sizes, coded_sizes + (int)(sizeof(int64_t) / (float)sizeof(T) * 3), data.get());
		cudaMemcpy(data.get() + coded_data_position_id_start, device_position_id, compression_blocks_count * sizeof(unsigned long), cudaMemcpyDeviceToHost);
		cudaMemcpy(data.get() + coded_data_bit_length_start, device_bit_length, compression_blocks_count * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaMemcpy(data.get() + host_out_start, device_compressed, compressed_data_size, cudaMemcpyDeviceToHost);

		cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_uncompressed), data_size);
		cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_compressed), data_size);
		cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_bit_length), compression_blocks_count * sizeof(unsigned char));
		cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_position_id), compression_blocks_count * sizeof(unsigned long));
		cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_compressed_size), sizeof(long));

		host_compressed.assign(data.get(), data.get() + compressed_size);

		result = true;
	}
	else
	{
		result = false;
	}

	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_uncompressed), data_size);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_compressed), data_size);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_bit_length), compression_blocks_count * sizeof(unsigned char));
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_position_id), compression_blocks_count * sizeof(unsigned long));
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_compressed_size), sizeof(long));

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
bool decompressAAFL(const int CWARP_SIZE, T* const host_compressed, int64_t compressed_size, std::vector<T>& host_uncompressed, T min, T max)
{
	T offset = min;

	int64_t data_size = reinterpret_cast<int64_t*>(host_compressed)[0];
	int64_t compressed_data_size = reinterpret_cast<int64_t*>(host_compressed)[1];
	int64_t compression_blocks_count = reinterpret_cast<int64_t*>(host_compressed)[2];

	int64_t size = data_size / sizeof(T);

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
	
	int coded_data_position_id_start = (sizeof(int64_t) / (float)sizeof(T) * 3);
	int coded_data_bit_length_start = coded_data_position_id_start + (sizeof(unsigned long) / (float)sizeof(T) * compression_blocks_count);
	int host_out_start = coded_data_bit_length_start + (sizeof(char) / (float)sizeof(T) * compression_blocks_count);

	host_position_id = reinterpret_cast<unsigned long*>(&host_compressed[coded_data_position_id_start]);
	host_bit_length = reinterpret_cast<unsigned char*>(&host_compressed[coded_data_bit_length_start]);
	host_compressed_data = &host_compressed[host_out_start];

	//for (int i = 0; i < 10; i++) {
	//	printf("bit2 %d\n", host_bit_length[i]);
	//}

	cudaMemcpy(device_compressed, host_compressed_data, compressed_data_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_position_id, host_position_id, compression_blocks_count * sizeof(unsigned long), cudaMemcpyHostToDevice);
	cudaMemcpy(device_bit_length, host_bit_length, compression_blocks_count * sizeof(unsigned char), cudaMemcpyHostToDevice);

	container_uncompressed<T> udata = { device_uncompressed, size };
	container_aafl<T> cdata = { device_compressed, size, device_bit_length, device_position_id, NULL, offset };

	gpu_fl_naive_launcher_decompression<T, 32, container_aafl<T>>::decompress(cdata, udata);
	
	std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[data_size / sizeof(T)]);
	cudaMemcpy(data.get(), device_uncompressed, data_size, cudaMemcpyDeviceToHost);
	
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_uncompressed), data_size);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_compressed), data_size);
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_bit_length), compression_blocks_count * sizeof(unsigned char));
	cudaAllocator.deallocate(reinterpret_cast<int8_t*>(device_position_id), compression_blocks_count * sizeof(unsigned long));
	
	host_uncompressed.reserve(size);
	host_uncompressed.assign(data.get(), data.get() + size);

	return true;
}

template<>
bool CompressionGPU::decompressDataAAFL<int32_t>(int32_t* const host_compressed, int64_t compressed_size, std::vector<int32_t>& host_uncompressed, int32_t min, int32_t max)
{
	return decompressAAFL(32, host_compressed, compressed_size, host_uncompressed, min, max);	
}

template<>
bool CompressionGPU::decompressDataAAFL<int64_t>(int64_t* const host_compressed, int64_t compressed_size, std::vector<int64_t>& host_uncompressed, int64_t min, int64_t max)
{
	return decompressAAFL(32, host_compressed, compressed_size, host_uncompressed, min, max);
}

template<>
bool CompressionGPU::decompressDataAAFL<int8_t>(int8_t* const host_compressed, int64_t compressed_size, std::vector<int8_t>& host_uncompressed, int8_t min, int8_t max)
{
	return decompressAAFL(32, host_compressed, compressed_size, host_uncompressed, min, max);
}

template<>
bool CompressionGPU::decompressDataAAFL<float>(float* const host_compressed, int64_t compressed_size, std::vector<float>& host_uncompressed, float min, float max)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFL<double>(double* const host_compressed, int64_t compressed_size, std::vector<double>& host_uncompressed, double min, double max)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFL<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon* const host_compressed, int64_t compressed_size, std::vector<ColmnarDB::Types::ComplexPolygon>& host_uncompressed, ColmnarDB::Types::ComplexPolygon min, ColmnarDB::Types::ComplexPolygon max)
{
	return false;
}

template<>
bool CompressionGPU::decompressDataAAFL<ColmnarDB::Types::Point>(ColmnarDB::Types::Point* const host_compressed, int64_t compressed_size, std::vector<ColmnarDB::Types::Point>& host_uncompressed, ColmnarDB::Types::Point min, ColmnarDB::Types::Point max)
{
	return false;
}






template<typename T>
bool decompressAAFLOnDevice(const int CWARP_SIZE, T* const device_compressed, int64_t data_size, int64_t compression_data_size, int64_t compression_blocks_count, T* const device_uncompressed, T min, T max)
{
	T offset = -1;

	int64_t size = data_size / sizeof(T);

	T *device_compressed_data;

	unsigned char *device_bit_length;
	unsigned long *device_position_id;
		
	int coded_data_position_id_start = (sizeof(int64_t) / (float)sizeof(T) * 3);
	int coded_data_bit_length_start = coded_data_position_id_start + std::max((int)(sizeof(unsigned long) / (float)sizeof(T) * compression_blocks_count), 1);
	int device_out_start = coded_data_bit_length_start + std::max((int)(sizeof(char) / (float)sizeof(T) * compression_blocks_count), 1);

	device_position_id = reinterpret_cast<unsigned long*>(&device_compressed[coded_data_position_id_start]);
	device_bit_length = reinterpret_cast<unsigned char*>(&device_compressed[coded_data_bit_length_start]);
	device_compressed_data = &device_compressed[device_out_start];

	
	container_uncompressed<T> udata = { device_uncompressed, size };
	container_aafl<T> cdata = { device_compressed_data, size, device_bit_length, device_position_id, NULL, offset };

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