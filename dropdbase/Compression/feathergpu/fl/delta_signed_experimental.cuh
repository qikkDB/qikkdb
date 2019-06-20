#pragma once
#include "../util/ptx.cuh"

template <typename T, char CWARP_SIZE>
__device__  void fl_compress_func ( unsigned long data_id, unsigned long comp_data_id, container_uncompressed<T> udata, container_signed_delta_fl<T> cdata)
{
    if (data_id >= udata.length) return;

    // TODO: Compressed data should be always unsigned, fix that latter
    T v1;
    unsigned int uv1;
    unsigned int value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;
    unsigned int sgn = 0;

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

        //TODO: ugly hack, fix that with correct bfe calls
        sgn = ((unsigned int) v1) >> 31;
        uv1 = abs(v1);
        // END: ugly hack

        if (v1_pos >= CWORD_SIZE(T) - cdata.bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            value = value | (GETNBITS(uv1, v1_len) << v1_pos);

            if (v1_pos == CWORD_SIZE(T) - cdata.bit_length) // whole word
                value |= (GETNBITS(uv1, v1_len - 1) | (sgn << (v1_len - 1))) << (v1_pos);
            else // begining of the word
                value |= GETNBITS(uv1, v1_len) << (v1_pos);

            cdata.data[pos] = reinterpret_cast<int&>(value);

            v1_pos = cdata.bit_length - v1_len;
            value = 0;
            // if is necessary as otherwise may work with negative bit shifts
            if (v1_pos > 0) // The last part of the word
                value = (GETNPBITS(uv1, v1_pos - 1, v1_len)) | (sgn << (v1_pos - 1));

            pos += CWARP_SIZE;
        } else {
            v1_len = cdata.bit_length;
            value |= (GETNBITS(uv1, v1_len-1) | (sgn << (v1_len-1))) << v1_pos;
            v1_pos += v1_len;
        }
    }

    if (pos_data >= udata.length  && pos_data < udata.length + CWARP_SIZE)
    {
        cdata.data[pos] = reinterpret_cast<int&>(value);
    }
}

template <typename T, char CWARP_SIZE>
__device__ void fl_decompress_func (unsigned long comp_data_id, unsigned long data_id, container_signed_delta_fl<T> cdata, container_uncompressed<T> udata)
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    unsigned int v1;
    unsigned int ret;
    int sret;

    const unsigned long lane = get_lane_id();

    if (pos_decomp >= udata.length ) // Decompress not more elements then length
        return;

    v1 = reinterpret_cast<unsigned int &>(cdata.data[pos]);

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
            v1 = reinterpret_cast<unsigned int &>(cdata.data[pos]);

            v1_pos = cdata.bit_length - v1_len;
            ret = ret | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = cdata.bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        // TODO: dirty hack
        int sgn_multiply = (ret >> (cdata.bit_length-1)) ? -1 : 1;
        // END
        ret &= NBITSTOMASK(cdata.bit_length-1);
        sret = sgn_multiply * (int)(ret);

        sret = shfl_prefix_sum(sret); // prefix sum deltas

        v2 = shfl_get_value(zeroLaneValue, 0);
        sret = v2 - sret;

        udata.data[pos_decomp] = sret;
        pos_decomp += CWARP_SIZE;

        v2 = shfl_get_value(sret, 31);

        if(lane == 0)
            zeroLaneValue = v2;
    }
}
