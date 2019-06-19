#pragma once
#include "afl.cuh"
#include "../util/ptx.cuh"

template <typename T, char CWARP_SIZE>
__device__  void fl_compress_func ( const unsigned long data_id, const unsigned long comp_data_id, container_uncompressed<T> udata, container_pafl<T> cdata)
{
    if (data_id >= udata.length) return;

    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;
    unsigned int exception_counter = 0;

    T exception_buffer[8];
    unsigned long position_mask = 0;
    T mask = ~BITMASK(T, cdata.bit_length);

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_data < udata.length; ++i)
    {
        v1 = udata.data[pos_data];

        if(v1 & mask){
            exception_buffer[exception_counter] = v1;
            exception_counter ++;
            BIT_SET(position_mask, i);
        }

        pos_data += CWARP_SIZE;

        if (v1_pos >= CWORD_SIZE(T) - cdata.bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);

            cdata.data[pos] = value;

            v1_pos = cdata.bit_length - v1_len;
            value = GETNPBITS(v1, v1_pos, v1_len);

            pos += CWARP_SIZE;
        } else {
            v1_len = cdata.bit_length;
            value = value | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if (pos_data >= udata.length  && pos_data < udata.length + CWARP_SIZE)
    {
        cdata.data[pos] = value;
    }

    unsigned int lane_id = get_lane_id();
    unsigned long local_counter = 0;

    unsigned int warp_exception_counter = shfl_prefix_sum(exception_counter);

    if(lane_id == 31 && warp_exception_counter > 0){
        local_counter = atomicAdd((unsigned long long int *)cdata.patch_count, (unsigned long long int)warp_exception_counter);
    }

    local_counter = shfl_get_value(local_counter, 31);

    for (unsigned int i = 0; i < exception_counter; ++i)
        cdata.patch_values[local_counter + warp_exception_counter - exception_counter + i] = exception_buffer [i];

    for (unsigned int i = 0, j = 0; i < exception_counter && j < CWORD_SIZE(T); j++){
        if (BIT_CHECK(position_mask, j)) {
            cdata.patch_index[local_counter + warp_exception_counter - exception_counter + i] = data_id + j * CWARP_SIZE;
            i++;
        }
    }
}

template <typename T, char CWARP_SIZE>
__global__ void patch_apply_kernel ( container_uncompressed<T> udata, container_pafl<T> cdata)
{
    unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned long patch_length = *cdata.patch_count;

    if (tid < patch_length)
    {
        unsigned long idx = cdata.patch_index[tid];
        T val = cdata.patch_values[tid];
        udata.data[idx] = val;
    }
}
