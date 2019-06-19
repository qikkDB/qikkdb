#pragma once
#include "../util/ptx.cuh"
#include "containers.cuh"

template <typename T, char CWARP_SIZE>
__device__  void fl_compress_func ( unsigned long data_id, unsigned long comp_data_id, container_uncompressed<T> udata, container_delta_fl<T> cdata)
{
    if (data_id >= udata.length) return;

    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;

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

        v2 = shfl_get_value((long)v1, neighborId);

        if (lane == 0)
        {
            // Lane 0 uses data from previous iteration
            v1 = zeroLaneValue - v1;
            zeroLaneValue = v2;
        } else {
            v1 = v2 - v1;
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
}

template <typename T, char CWARP_SIZE>
__device__ void fl_decompress_func (unsigned long comp_data_id, unsigned long data_id, container_delta_fl<T> cdata, container_uncompressed<T> udata)
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

        ret = shfl_prefix_sum((long)ret); // prefix sum deltas
        v2 = shfl_get_value((long)zeroLaneValue, 0);
        ret = v2 - ret;

        udata.data[pos_decomp] = ret;
        pos_decomp += CWARP_SIZE;

        v2 = shfl_get_value((long)ret, 31);

        if(lane == 0)
            zeroLaneValue = v2;
    }
}

