#pragma once
#include "afl.cuh"

template <typename T, char CWARP_SIZE>
__device__  void fl_compress_func (unsigned long data_id, container_uncompressed<T> udata, container_aafl<T> cdata)
{

    unsigned long pos_data = data_id;

    unsigned int bit_length = 0, i = 0;
    const unsigned int warp_lane = get_lane_id(CWARP_SIZE);

    unsigned long long max_val = 0;

    // Compute bit length for compressed block of data
    for (i = 0; i < CWORD_SIZE(T) && pos_data < udata.length; ++i)
    {
        max_val = MAKE_UNSIGNED(udata.data[pos_data]) > MAKE_UNSIGNED(max_val) ? MAKE_UNSIGNED(udata.data[pos_data]) : MAKE_UNSIGNED(max_val);
        pos_data += CWARP_SIZE;		
    }

    i = warpAllReduceMax(i);
    // Warp vote for maximum bit length
    bit_length = max_val > 0 ? BITLEN(max_val) + 1 : 0;
    bit_length = warpAllReduceMax(bit_length);

    // leader thread registers memory in global
    unsigned long comp_data_id = 0;

    if (warp_lane == 0) {
        const unsigned long data_block = (blockIdx.x * blockDim.x) / CWARP_SIZE + threadIdx.x / CWARP_SIZE;
        unsigned long long int space = bit_length * CWARP_SIZE;

        if(data_id + CWARP_SIZE * CWORD_SIZE(T) > udata.length && data_id < udata.length) { // We process data in blocks of N elements, this is needed if data size is not a multiply of N
            space = (( (udata.length - data_id + CWORD_SIZE(T) - 1) / CWORD_SIZE(T)) * bit_length + CWARP_SIZE - 1) / CWARP_SIZE;
            space *= CWARP_SIZE;
        }

        comp_data_id = (unsigned long long int) atomicAdd( (unsigned long long int *) cdata.data_register, space);
        cdata.warp_bit_lenght[data_block] = bit_length;
        cdata.warp_position_id[data_block] = comp_data_id;
    }
	
    if (bit_length > 0) { // skip if bit_length is 0 for whole block (i.e. all values are equal 0)
        // Propagate in warp position of compressed block
        comp_data_id = warpAllReduceMax(comp_data_id);
        comp_data_id += warp_lane;

        // Compress using AFL algorithm
        container_fl<T> cdata_fl = {(unsigned char) bit_length, cdata.data, cdata.length};
        fl_compress_func <T, CWARP_SIZE> (data_id, comp_data_id, udata, cdata_fl);
    }
}

template < typename T, char CWARP_SIZE >
__global__ void gpu_aafl_compress_kernel ( container_uncompressed<T> udata, container_aafl<T> cdata)
{
    const unsigned long data_id = get_data_id <T,CWARP_SIZE> ();
    fl_compress_func <T, CWARP_SIZE> (data_id, udata, cdata);
}

template < typename T, char CWARP_SIZE >
__global__ void gpu_aafl_decompress_kernel ( container_aafl<T> cdata, container_uncompressed<T> udata)
{
    const unsigned long data_id = get_data_id <T, CWARP_SIZE> ();

    if (data_id >= cdata.length) return;

    const unsigned long data_block_mem = (blockIdx.x * blockDim.x) / CWARP_SIZE  + threadIdx.x / CWARP_SIZE;
    unsigned long comp_data_id = cdata.warp_position_id[data_block_mem] + get_lane_id();
    unsigned int bit_length = cdata.warp_bit_lenght[data_block_mem];

    container_fl<T> cdata_fl = {(unsigned char) bit_length, cdata.data, cdata.length};

    if(bit_length > 0)
        fl_decompress_func <T, CWARP_SIZE> (comp_data_id, data_id, cdata_fl, udata);
    else
        afl_decompress_constant_value <T, CWARP_SIZE> (comp_data_id, data_id, cdata_fl, udata, 0);
}


