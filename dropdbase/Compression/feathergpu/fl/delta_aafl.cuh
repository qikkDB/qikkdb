#pragma once
#include "delta.cuh"
template <typename T, char CWARP_SIZE>
__device__  void fl_compress_func ( unsigned long data_id, container_uncompressed<T> udata, container_delta_aafl<T> cdata)
{

    unsigned long pos_data = data_id;
    unsigned int bit_length = 0, i = 0;
    const unsigned long lane = get_lane_id();
    char neighborId = lane - 1;
    T zeroLaneValue, v1, v2, block_start;
    T max_val = 0;

    if (lane == 0 )  {
        neighborId = 31;
        zeroLaneValue = udata.data[pos_data];
        block_start = zeroLaneValue;
    }

    // Compute bit length for compressed block of data
    for (i = 0; i < CWORD_SIZE(T) && pos_data < udata.length; ++i)
    {

        v1 = udata.data[pos_data];
        pos_data += CWARP_SIZE;

        v2 = shfl_get_value(v1, neighborId);

        if (lane == 0)
        {
            // Lane 0 uses data from previous iteration
            v1 = zeroLaneValue - v1;
            zeroLaneValue = v2;
        } else {
            v1 = v2 - v1;
        }

        max_val = v1 > max_val ?  v1 : max_val;
    }

    i = warpAllReduceMax(i);
    // Warp vote for maximum bit length
    bit_length = max_val > 0 ? BITLEN(max_val) + 1 : 0;
    bit_length = warpAllReduceMax(bit_length);

    // leader thread registers memory in global
    unsigned long comp_data_id = 0;

    if (lane == 0) {
        const unsigned long data_block = (blockIdx.x * blockDim.x) / CWARP_SIZE + threadIdx.x / CWARP_SIZE;
        unsigned long long int space = bit_length * CWARP_SIZE;

        if(data_id + CWARP_SIZE * CWORD_SIZE(T) > udata.length && data_id < udata.length) {
            space = (( (udata.length - data_id + CWORD_SIZE(T) - 1) / CWORD_SIZE(T)) * bit_length + CWARP_SIZE - 1) / CWARP_SIZE;
            space *= CWARP_SIZE;
        }

        comp_data_id = (unsigned long long int) atomicAdd( (unsigned long long int *) cdata.data_register, space);
        cdata.warp_bit_lenght[data_block] = bit_length;
        cdata.warp_position_id[data_block] = comp_data_id;
        cdata.block_start[data_block] = block_start;
    }

    if (bit_length > 0) {
        // Propagate in warp position of compressed block
        comp_data_id = warpAllReduceMax(comp_data_id);
        comp_data_id += lane;

        // Compress using AFL algorithm
        container_delta_fl<T> cdata_delta_fl = {(unsigned char) bit_length, (T *) cdata.data, cdata.length, (T *) cdata.block_start};
        fl_compress_func <T, CWARP_SIZE> (data_id, comp_data_id, udata, cdata_delta_fl);
    }
}

template < typename T, char CWARP_SIZE >
__global__ void gpu_delta_aafl_compress_kernel ( container_uncompressed<T> udata, container_delta_aafl<T> cdata)
{
    const unsigned long data_id = get_data_id <T, CWARP_SIZE> ();
    fl_compress_func <T, CWARP_SIZE> (data_id, udata, cdata);
}

template < typename T, char CWARP_SIZE >
__global__ void gpu_delta_aafl_decompress_kernel ( container_delta_aafl<T> cdata, container_uncompressed<T> udata)
{
    const unsigned long data_id = get_data_id <T, CWARP_SIZE> ();

    if (data_id >= udata.length) return;

    const unsigned long data_block_mem = (blockIdx.x * blockDim.x) / CWARP_SIZE  + threadIdx.x / CWARP_SIZE;
    unsigned long comp_data_id = cdata.warp_position_id[data_block_mem] + get_lane_id();
    unsigned int bit_length = cdata.warp_bit_lenght[data_block_mem];

    if(bit_length > 0){
        container_delta_fl<T> cdata_delta_fl = {(unsigned char) bit_length, (T *) cdata.data, udata.length, (T *) cdata.block_start};
        fl_decompress_func <T, CWARP_SIZE> (comp_data_id, data_id, cdata_delta_fl, udata);
    } else {
        container_fl<T> cdata_fl = {(unsigned char) bit_length, (T *) cdata.data, udata.length};

        afl_decompress_constant_value <T, CWARP_SIZE> (comp_data_id, data_id, cdata_fl, udata, cdata.block_start[data_block_mem]);
    }
}

