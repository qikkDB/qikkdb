#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helpers.cuh"
#include "afl.cuh"
#include "aafl.cuh"
#include "pafl.cuh"
#include "delta_pafl.cuh"
#include "delta_aafl.cuh"
#include "afl_signed_experimental.cuh"
#include "delta_signed_experimental.cuh"
#include "../util/cuda.cuh"

template < typename T, char CWARP_SIZE, typename CCONT>
__global__ void gpu_default_decompress_kernel (CCONT cdata, container_uncompressed<T> udata)
{
    unsigned long data_id, cdata_id;
    set_cmp_offset <T, CWARP_SIZE> (threadIdx.x, blockIdx.x * blockDim.x, cdata.bit_length, data_id, cdata_id);

    fl_decompress_func <T, CWARP_SIZE> (cdata_id, data_id, cdata, udata);
}

template < typename T, char CWARP_SIZE , typename CCONT>
__global__ void gpu_default_compress_kernel (container_uncompressed<T> udata, CCONT cdata)
{
    unsigned long data_id, cdata_id;
    set_cmp_offset <T, CWARP_SIZE> (threadIdx.x, blockIdx.x * blockDim.x, cdata.bit_length, data_id, cdata_id);

    fl_compress_func <T, CWARP_SIZE> (data_id, cdata_id, udata, cdata);
}

template < typename T, char CWARP_SIZE , typename CCONT>
struct gpu_fl_naive_launcher_compression {
    __host__ static void compress (container_uncompressed<T> udata, CCONT cdata)
    {
        const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        const unsigned long block_number = (udata.length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

        gpu_default_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
    }
};

template < typename T, char CWARP_SIZE , typename CCONT>
struct gpu_fl_naive_launcher_decompression {

    __host__ static void decompress (CCONT cdata, container_uncompressed<T> udata)
    {
        const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        const unsigned long block_number = (udata.length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

        gpu_default_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
    }
};

//AAFL specialization
template < typename T, char CWARP_SIZE>
struct gpu_fl_naive_launcher_compression <T, CWARP_SIZE, container_aafl<T>>{
    __host__ static void compress (container_uncompressed<T> udata, container_aafl<T> cdata)
    {
        const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        const unsigned long block_number = (udata.length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

        gpu_aafl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
    }
};
template < typename T, char CWARP_SIZE>
struct gpu_fl_naive_launcher_decompression <T, CWARP_SIZE, container_aafl<T>>{
    __host__ static void decompress (container_aafl<T> cdata, container_uncompressed<T> udata)
    {
        const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        const unsigned long block_number = (udata.length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

        gpu_aafl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
    }
};
//DELTA-AAFL specialization
template < typename T, char CWARP_SIZE>
struct gpu_fl_naive_launcher_compression <T, CWARP_SIZE, container_delta_aafl<T>>{
    __host__ static void compress (container_uncompressed<T> udata, container_delta_aafl<T> cdata)
    {
        const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        const unsigned long block_number = (udata.length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

        gpu_delta_aafl_compress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (udata, cdata);
    }
};
template < typename T, char CWARP_SIZE>
struct gpu_fl_naive_launcher_decompression <T, CWARP_SIZE, container_delta_aafl<T>>{
    __host__ static void decompress (container_delta_aafl<T> cdata, container_uncompressed<T> udata)
    {
        const unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        const unsigned long block_number = (udata.length + block_size * CWORD_SIZE(T) - 1) / (block_size * CWORD_SIZE(T));

        gpu_delta_aafl_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
    }
};

//PAFL specialization
template < typename T, char CWARP_SIZE>
struct gpu_fl_naive_launcher_decompression <T, CWARP_SIZE, container_pafl<T>>{

    __host__ static void decompress (container_pafl<T> cdata, container_uncompressed<T> udata)
    {
        container_fl<T> cdata_fl = { cdata.bit_length, cdata.data, cdata.length};
        gpu_fl_naive_launcher_decompression<T, CWARP_SIZE, container_fl<T>>::decompress(cdata_fl, udata);
        cudaErrorCheck();
        unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        unsigned long block_number = (cdata.length + block_size * CWARP_SIZE - 1) / (block_size * CWARP_SIZE);

        patch_apply_kernel <T, CWARP_SIZE> <<<block_number * CWARP_SIZE, block_size>>> (udata, cdata);
        cudaErrorCheck();
    }
};

//PAFL specialization
template < typename T, char CWARP_SIZE>
struct gpu_fl_naive_launcher_decompression <T, CWARP_SIZE, container_delta_pafl<T>>{

    __host__ static void decompress (container_delta_pafl<T> cdata, container_uncompressed<T> udata)
    {
        unsigned int block_size = CWARP_SIZE * 8; // better occupancy
        unsigned long block_number = (cdata.length + block_size * CWARP_SIZE - 1) / (block_size * CWARP_SIZE);

        container_pafl<T> cdata_pafl = {cdata.bit_length, cdata.data, cdata.length, cdata.patch_values, cdata.patch_index, cdata.patch_count};

        patch_apply_kernel <T, CWARP_SIZE> <<<block_number * CWARP_SIZE, block_size>>> (udata, cdata_pafl);
        cudaErrorCheck();

        gpu_default_decompress_kernel <T, CWARP_SIZE> <<<block_number, block_size>>> (cdata, udata);
        cudaErrorCheck();
    }
};



// Launchers
template < typename T, char CWARP_SIZE, typename X>
__host__ void compress (container_uncompressed<T> udata, X cdata)
{
    gpu_fl_naive_launcher_compression<T, CWARP_SIZE, X>::compress(udata, cdata);
}

template < typename T, char CWARP_SIZE, typename X>
__host__ void decompress (X cdata, container_uncompressed<T> udata)
{
    gpu_fl_naive_launcher_decompression<T, CWARP_SIZE, X>::decompress(cdata, udata);
}
