#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "dropdbase/Compression/feathergpu/fl/containers.cuh"
#include "dropdbase/Compression/feathergpu/fl/default.cuh"
#include "GPUCompression.h"
#include "dropdbase/QueryEngine/Context.h" 
#include <memory>
#include <string>
#include "dropdbase/Types/ComplexPolygon.pb.h"
#include "dropdbase/Types/Point.pb.h"


template<typename T>
bool compressAAFL(const int CWARP_SIZE, T* const host_uncompressed, int64_t size, std::vector<T>& host_compressed, int64_t& compressed_size)
{
	int64_t data_size = size * sizeof(T);
	int64_t compression_blocks_count = (data_size + (sizeof(T) * CWARP_SIZE) - 1) / (sizeof(T) * CWARP_SIZE);

	//T *host_compressed;
	unsigned char *host_bit_length;
	unsigned long *host_position_id;

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
	container_aafl<T> cdata = { device_compressed, size, device_bit_length, device_position_id, device_compressed_size };
	gpu_fl_naive_launcher_compression<T, 32, container_aafl<T>>::compress(udata, cdata);

	unsigned long host_compressed_size;
	cudaMemcpy(&host_compressed_size, device_compressed_size, sizeof(unsigned long), cudaMemcpyDeviceToHost);
	int64_t compressed_data_size = (host_compressed_size) * sizeof(T);

	// coding into single array
	unsigned long compressed_data_size_final = compressed_data_size + (sizeof(unsigned long) * compression_blocks_count) + (sizeof(unsigned char) * compression_blocks_count) + (sizeof(int64_t) * 3);
	compressed_size = compressed_data_size_final / sizeof(T);
	if (compressed_size < size)
	{
		int64_t sizes[3] = { data_size , compressed_data_size, compression_blocks_count };

		T* coded_sizes = reinterpret_cast<T*>(sizes);

		int coded_data_position_id_start = (sizeof(int64_t) / sizeof(T) * 3);
		int coded_data_bit_length_start = coded_data_position_id_start + (sizeof(unsigned long) / sizeof(T) * compression_blocks_count);
		int host_out_start = coded_data_bit_length_start + (sizeof(char) / sizeof(T) * compression_blocks_count);

		
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
		return 1;
	}
	else
	{
		return 0;
	}
}





template<>
bool CompressionGPU::compressDataAAFL<int32_t>(int32_t* const host_uncompressed, int64_t size, std::vector<int32_t>& host_compressed, int64_t& compressed_size)
{
	compressAAFL(32, host_uncompressed, size, host_compressed, compressed_size);
	return 1;
}

template<>
bool CompressionGPU::compressDataAAFL<int64_t>(int64_t* const host_uncompressed, int64_t size, std::vector<int64_t>& host_compressed, int64_t& compressed_size)
{
	compressAAFL(32, host_uncompressed, size, host_compressed, compressed_size);
	return 1;
}

template<>
bool CompressionGPU::compressDataAAFL<int8_t>(int8_t* const host_uncompressed, int64_t size, std::vector<int8_t>& host_compressed, int64_t& compressed_size)
{
	compressAAFL(32, host_uncompressed, size, host_compressed, compressed_size);
	return 1;
}

template<>
bool CompressionGPU::compressDataAAFL<double>(double* const host_uncompressed, int64_t size, std::vector<double>& host_compressed, int64_t& compressed_size)
{	
	return 0;
}

template<>
bool CompressionGPU::compressDataAAFL<float>(float* const host_uncompressed, int64_t size, std::vector<float>& host_compressed, int64_t& compressed_size)
{
	return 0;
}

template<>
bool CompressionGPU::compressDataAAFL<std::string>(std::string* const host_uncompressed, int64_t size, std::vector<std::string>& host_compressed, int64_t& compressed_size)
{
	return 0;
}


template<>
bool CompressionGPU::compressDataAAFL<ColmnarDB::Types::ComplexPolygon>(ColmnarDB::Types::ComplexPolygon* const host_uncompressed, int64_t size, std::vector<ColmnarDB::Types::ComplexPolygon>& host_compressed, int64_t& compressed_size)
{
	return 0;
}

template<>
bool CompressionGPU::compressDataAAFL<ColmnarDB::Types::Point>(ColmnarDB::Types::Point* const host_uncompressed, int64_t size, std::vector<ColmnarDB::Types::Point>& host_compressed, int64_t& compressed_size)
{
	return 0;
}


