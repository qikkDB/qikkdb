#pragma once
#include "afl.cuh"


template < typename T, char CWARP_SIZE >
__global__ void gpu_aafl_compact_decompress_kernel(T* cdata, container_uncompressed<T> udata)
{
	const unsigned long data_id = get_data_id <T, CWARP_SIZE>();

	

	// decode input
	long long data_size = reinterpret_cast<long long*>(cdata)[0];
	long long compressed_data_size = reinterpret_cast<long long*>(cdata)[1];
	long long compression_blocks_count = reinterpret_cast<long long*>(cdata)[2];

	//printf("data_size %lld\n", data_size);
	//printf("compressed_data_size %lld\n", compressed_data_size);
	//printf("compression_blocks_count %lld\n", compression_blocks_count);

	unsigned long size = data_size / sizeof(T);
	
	if (data_id >= udata.length) return;
	
	int coded_data_position_id_start = (sizeof(long long) / (float)sizeof(T) * 3);
	int coded_data_bit_length_start = coded_data_position_id_start + (sizeof(unsigned long) / (float)sizeof(T) * compression_blocks_count);
	int coded_out_start = coded_data_bit_length_start + (sizeof(char) / (float)sizeof(T) * compression_blocks_count);

	unsigned long* host_position_id = reinterpret_cast<unsigned long*>(&cdata[coded_data_position_id_start]);
	unsigned char* host_bit_length = reinterpret_cast<unsigned char*>(&cdata[coded_data_bit_length_start]);
	T* host_compressed = &cdata[coded_out_start];
	//--


	const unsigned long data_block_mem = (blockIdx.x * blockDim.x) / CWARP_SIZE + threadIdx.x / CWARP_SIZE;
	unsigned long comp_data_id = host_position_id[data_block_mem] + get_lane_id();
	unsigned char bit_length = host_bit_length[data_block_mem];

	container_fl<T> cdata_fl = { (unsigned char)bit_length, host_compressed, udata.length };
	
	if (bit_length > 0)
		fl_decompress_func <T, CWARP_SIZE>(comp_data_id, data_id, cdata_fl, udata);
	else
		afl_decompress_constant_value <T, CWARP_SIZE>(comp_data_id, data_id, cdata_fl, udata, 0);
	//if (comp_data_id < 10)
	//	printf("data_id %d\n", host_compressed[comp_data_id]);
	
}


