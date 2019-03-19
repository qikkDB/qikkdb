#pragma once
#include "../util/ptx.cuh"
#include "containers.cuh"
#include "helpers.cuh"

template <typename T, char CWARP_SIZE>
__device__  __host__ void fl_compress_func (unsigned long data_id, unsigned long comp_data_id, container_uncompressed<T> udata, container_fl<T> cdata)
{
    T v1, value = 0;
    unsigned int v1_pos=0, v1_len;
    unsigned long pos=comp_data_id, pos_data=data_id;

    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_data < udata.length; ++i)
    {
        v1 = udata.data[pos_data];
        pos_data += CWARP_SIZE;

        if (v1_pos >= CWORD_SIZE(T) - cdata.bit_length){
            // Word does not fit whole into the free space or fits but fills all empty space
            v1_len = CWORD_SIZE(T) - v1_pos;
            // Write part of the word (or whole word in case it would use all free space)
            value = MAKE_UNSIGNED(value) | (GETNBITS(v1, v1_len) << v1_pos);

            // flush compression buffer
            cdata.data[pos] = value;

            // write rest of the word into "new" buffer
            v1_pos = cdata.bit_length - v1_len;
            // if v1_pos ==0 then this sets value = 0, this allows us to skip if instruction
            value = GETNPBITS(v1, v1_pos, v1_len);

            pos += CWARP_SIZE;
        } else {
            // compressed word fits whole into free space and there is still some free space left
            v1_len = cdata.bit_length;
            value = MAKE_UNSIGNED(value) | (GETNBITS(v1, v1_len) << v1_pos);
            v1_pos += v1_len;
        }
    }
    if (pos_data >= udata.length  && pos_data < udata.length + CWARP_SIZE)
    {
        // In case there are buffers not entirely filled, but there are no additional values to compress
        // flush the buffers
        cdata.data[pos] = value;
    }
}

template <typename T, char CWARP_SIZE>
__device__ __host__ void fl_decompress_func (unsigned long comp_data_id, unsigned long data_id, container_fl<T> cdata, container_uncompressed<T> udata)
{
    unsigned long pos = comp_data_id, pos_decomp = data_id;
    unsigned int v1_pos = 0, v1_len;
    T v1, ret;

    if (pos_decomp > cdata.length ) // Decompress not more elements then length
        return;
    v1 = cdata.data[pos];
    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_decomp < cdata.length; ++i)
    {
        if (v1_pos >= CWORD_SIZE(T) - cdata.bit_length){
            v1_len = CWORD_SIZE(T) - v1_pos;
            ret = GETNPBITS(v1, v1_len, v1_pos);

            pos += CWARP_SIZE;
            v1 = cdata.data[pos];

            v1_pos = cdata.bit_length - v1_len;
            ret = MAKE_UNSIGNED(ret) | ((GETNBITS(v1, v1_pos))<< v1_len);
        } else {
            v1_len = cdata.bit_length;
            ret = GETNPBITS(v1, v1_len, v1_pos);
            v1_pos += v1_len;
        }

        udata.data[pos_decomp] = ret;
        pos_decomp += CWARP_SIZE;
    }
}

template <typename T, char CWARP_SIZE>
__device__ __host__ void afl_decompress_constant_value (unsigned long comp_data_id, unsigned long data_id, container_fl<T> cdata, container_uncompressed<T> udata, T value)
{
    unsigned long pos_decomp = data_id;

    if (pos_decomp > cdata.length ) // Decompress not more elements then length
        return;
    for (unsigned int i = 0; i < CWORD_SIZE(T) && pos_decomp < cdata.length; ++i)
    {
        udata.data[pos_decomp] = value;
        pos_decomp += CWARP_SIZE;
    }
}

template <typename T, char CWARP_SIZE>
__device__ __host__ T afl_decompress_value (container_fl<T> cdata, unsigned long pos)
{
    //TODO: check if we do not read data from pos > length
    const unsigned int data_block = pos / (CWARP_SIZE * CWORD_SIZE(T));
    const unsigned int pos_in_block = (pos % (CWARP_SIZE * CWORD_SIZE(T)));
    const unsigned int pos_in_warp_lane = pos_in_block % CWARP_SIZE;
    const unsigned int pos_in_warp_comp_block = pos_in_block / CWARP_SIZE;

    const unsigned long cblock_id =
        data_block * ( CWARP_SIZE * cdata.bit_length)
        + pos_in_warp_lane
        + ((pos_in_warp_comp_block * cdata.bit_length) / CWORD_SIZE(T)) * CWARP_SIZE;

    const unsigned int bit_pos = pos_in_warp_comp_block * cdata.bit_length % CWORD_SIZE(T);
    const unsigned int bit_ret = bit_pos <= CWORD_SIZE(T)  - cdata.bit_length  ? cdata.bit_length : CWORD_SIZE(T) - bit_pos;

    T ret = GETNPBITS(cdata.data[cblock_id], bit_ret, bit_pos);

    if (bit_ret < cdata.bit_length)
        ret |= GETNBITS(cdata.data[cblock_id+CWARP_SIZE], cdata.bit_length - bit_ret) << bit_ret;

    return ret;
}

template <typename T, char CWARP_SIZE>
__device__ __host__ void afl_compress_value ( container_fl<T> cdata, unsigned long pos, T value)
{
    //TODO: For GPU it requires atomic writes
    const unsigned int data_block = pos / (CWARP_SIZE * CWORD_SIZE(T));
    const unsigned int pos_in_block = (pos % (CWARP_SIZE * CWORD_SIZE(T)));
    const unsigned int pos_in_warp_lane = pos_in_block % CWARP_SIZE;
    const unsigned int pos_in_warp_comp_block = pos_in_block / CWARP_SIZE;

    const unsigned long cblock_id =
        data_block * ( CWARP_SIZE * cdata.bit_length) // move to data block
        + pos_in_warp_lane // move to starting position in data block
        + ((pos_in_warp_comp_block * cdata.bit_length) / CWORD_SIZE(T)) * CWARP_SIZE; // move to value

    const unsigned int bit_pos = pos_in_warp_comp_block * cdata.bit_length % CWORD_SIZE(T);
    const unsigned int bit_ret = bit_pos <= CWORD_SIZE(T)  - cdata.bit_length  ? cdata.bit_length : CWORD_SIZE(T) - bit_pos;


    SETNPBITS((T *)(cdata.data + cblock_id), value, bit_ret, bit_pos);

    if (bit_ret < cdata.bit_length)
        SETNPBITS((T*)(cdata.data + cblock_id + CWARP_SIZE), value >> bit_ret, cdata.bit_length - bit_ret, 0);
}

template < typename T, char CWARP_SIZE >
__global__ void afl_decompress_value_kernel (container_fl<T> cdata, container_uncompressed<T> udata)
{
    const unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < cdata.length)
    {
        udata.data[tid] = afl_decompress_value <T, CWARP_SIZE> (cdata, tid);
    }
}

template < typename T, char CWARP_SIZE >
__host__ void afl_compress_value_cpu_kernel(container_uncompressed<T> udata, container_fl<T> cdata)
{
    unsigned long tid;

    for (tid = 0; tid < udata.length; tid++)
        afl_compress_value <T, CWARP_SIZE> (cdata, tid, udata.data[tid]);
}

template < typename T, char CWARP_SIZE >
__host__ void afl_decompress_value_cpu_kernel(container_fl<T> cdata, container_uncompressed<T> udata)
{
    unsigned long tid;

    for (tid = 0; tid < udata.length; tid++)
        udata.data[tid] = afl_decompress_value <T, CWARP_SIZE> (cdata, tid);
}

template < typename T, char CWARP_SIZE >
__host__ void afl_compress_cpu_kernel( container_uncompressed<T> udata, container_fl<T> cdata)
{
    const unsigned int block_size = CWARP_SIZE * 8;
    const unsigned long block_number = (udata.length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    unsigned long tid, bid;

    for (tid = 0, bid = 0; bid < block_number; tid++)
    {
        if (tid == block_size)
        {
           tid = 0;
           bid += 1;
        }

        unsigned long data_id, cdata_id;
        set_cmp_offset<T,CWARP_SIZE>(tid, bid * block_size, cdata.bit_length, data_id, cdata_id);

        fl_compress_func <T, CWARP_SIZE> (data_id, cdata_id, udata, cdata);
    }
}

template < typename T, char CWARP_SIZE >
__host__ void afl_decompress_cpu_kernel(container_fl<T> cdata, container_uncompressed<T> udata)
{
    const unsigned int block_size = CWARP_SIZE * 8;
    const unsigned long block_number = (cdata.length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

    unsigned long tid, bid;

    for (tid = 0, bid = 0; bid <= block_number; tid++)
    {
        if (tid == block_size)
        {
           tid = 0;
           bid += 1;
        }

        unsigned long data_id, cdata_id;
        set_cmp_offset<T,CWARP_SIZE>(tid, bid * block_size, cdata.bit_length, data_id, cdata_id);

        fl_decompress_func <T, CWARP_SIZE> (cdata_id, data_id, cdata, udata);
    }
}

