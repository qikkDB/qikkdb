#pragma once
#include "../util/ptx.cuh"

template <typename T, char CWARP_SIZE>
__device__  void fl_compress_func ( const unsigned long data_id, const unsigned long comp_data_id, container_uncompressed<T> udata, container_delta_pafl<T> cdata)
{
    if (data_id >= udata.length) return;

    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;
    unsigned int exception_counter = 0;

    T exception_buffer[8];
    unsigned long position_mask = 0;
    T mask = ~BITMASK(T, cdata.bit_length);

    T zeroLaneValue, v2;
    const unsigned long lane = get_lane_id();
    char neighborId = lane - 1;

    const unsigned long data_block = ( blockIdx.x * blockDim.x) / CWARP_SIZE + threadIdx.x / CWARP_SIZE;

    if (lane == 0 )  {
        neighborId = 31;
        zeroLaneValue = udata.data[pos_data];
        cdata.block_start[data_block] = zeroLaneValue;
    }

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_data < udata.length; ++i)
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

        if(v1 & mask){
            exception_buffer[exception_counter] = v1;
            exception_counter ++;
            BIT_SET(position_mask, i);
        }

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

    unsigned int warp_exception_counter = shfl_prefix_sum((int)exception_counter);

    if(lane_id == 31 && warp_exception_counter > 0){
        local_counter = atomicAdd((unsigned long long int *)cdata.patch_count, (unsigned long long int)warp_exception_counter);
    }

    local_counter = shfl_get_value((long)local_counter, 31);

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
__device__ void fl_decompress_func ( unsigned long comp_data_id, unsigned long data_id, container_delta_pafl<T> cdata, container_uncompressed<T> udata)
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    T v1, ret;

    const unsigned long lane = get_lane_id();

    if (pos_decomp >= udata.length ) // Decompress not more elements then length
        return;

    v1 = cdata.data[pos];

    T zeroLaneValue = 0, v2 = 0;

    const unsigned long data_block = (blockIdx.x * blockDim.x) / CWARP_SIZE  + threadIdx.x / CWARP_SIZE;

    if (lane == 0) {
       zeroLaneValue = cdata.block_start[data_block];
    }

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_decomp < udata.length; ++i)
    {
        if (v1_pos >= CWORD_SIZE(T) - cdata.bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += CWARP_SIZE;
            v1 = cdata.data[pos];

            v1_pos = cdata.bit_length - v1_len;
            ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = cdata.bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        // if output array is not empty, we assume that exceptions were unpacked there
        if(udata.data[pos_decomp] > 0)
            ret = udata.data[pos_decomp];

        ret = shfl_prefix_sum(ret); // prefix sum deltas
        v2 = shfl_get_value(zeroLaneValue, 0);
        ret = v2 - ret;

        udata.data[pos_decomp] = ret;
        pos_decomp += CWARP_SIZE;

        v2 = shfl_get_value(ret, 31);

        if(lane == 0)
            zeroLaneValue = v2;
    }
}
