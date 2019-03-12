#pragma once

#include <cstdio>
#include <chrono>
#include <stdexcept>
#include <stdint.h>
#include <inttypes.h>
#include "../BlockBase.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "feathergpu/fl/containers.cuh"
#include "dropdbase/Compression/feathergpu/fl/default.cuh"

//#include "feathergpu/fl/aafl_compact.cuh"



class CompressionGPU
{


public:
    static const char CWARP_SIZE = 32;

    template<typename T>
    static std::unique_ptr<T[]> compressDataAAFL(T* const host_uncompressed, int64_t size, int64_t& compressed_size)
    {
        int64_t data_size = size * sizeof(T);
        int64_t compression_blocks_count = (data_size + (sizeof(T) * CWARP_SIZE) - 1) / (sizeof(T) * CWARP_SIZE);

        T *host_compressed;
        unsigned char *host_bit_length;
        unsigned long *host_position_id;

        T *device_uncompressed;
        T *device_compressed;
        unsigned char *device_bit_length;
        unsigned long *device_position_id;
        unsigned long *device_compressed_size;

        //allocations
        cudaMalloc(&device_uncompressed, data_size);
        cudaMalloc(&device_compressed, data_size); // first we do not know what will be the size, therfore data_size

        cudaMalloc(&device_bit_length, compression_blocks_count * sizeof(unsigned char));
        cudaMalloc(&device_position_id, compression_blocks_count * sizeof(unsigned long));
        cudaMalloc(&device_compressed_size, sizeof(long));

        //copy M->G
        cudaMemcpy(device_uncompressed, host_uncompressed, data_size, cudaMemcpyHostToDevice);

        // Clean up before compression
        cudaMemset(device_compressed, 0, data_size);
        cudaMemset(device_compressed_size, 0, sizeof(unsigned long));
        cudaMemset(device_bit_length, 0, compression_blocks_count * sizeof(unsigned char));
        cudaMemset(device_position_id, 0, compression_blocks_count * sizeof(unsigned long));

        container_uncompressed<T> udata = { device_uncompressed, size };
        container_aafl<T> cdata = { device_compressed, size, device_bit_length, device_position_id, device_compressed_size };
        gpu_fl_naive_launcher_compression<T, CWARP_SIZE, container_aafl<T>>::compress(udata, cdata);

        unsigned long host_compressed_size;
        cudaMemcpy(&host_compressed_size, device_compressed_size, sizeof(unsigned long), cudaMemcpyDeviceToHost);
        int64_t compressed_data_size = (host_compressed_size) * sizeof(T);


        unsigned long compressed_data_size_final = compressed_data_size + (sizeof(unsigned long)/(float)sizeof(T) * compression_blocks_count) + (sizeof(unsigned char) / (float)sizeof(T) * compression_blocks_count) + (sizeof(int64_t) / (float)sizeof(T) * 3);

        int64_t sizes[3] = { data_size , compressed_data_size, compression_blocks_count };

        T* coded_sizes = reinterpret_cast<T*>(sizes);

        int coded_data_position_id_start = (sizeof(int64_t) / (float)sizeof(T) * 3);
        int coded_data_bit_length_start = coded_data_position_id_start + (sizeof(unsigned long) / (float)sizeof(T) * compression_blocks_count);
        int host_out_start = coded_data_bit_length_start + (sizeof(char) / (float)sizeof(T) * compression_blocks_count);

        std::unique_ptr<T[]> data = std::unique_ptr<T[]>(new T[compressed_data_size_final]);
        std::move(coded_sizes, coded_sizes + (int)(sizeof(int64_t) / (float)sizeof(T) * 3), data.get());
        cudaMemcpy(data.get() + coded_data_position_id_start, device_position_id, compression_blocks_count * sizeof(unsigned long), cudaMemcpyDeviceToHost);

        cudaMemcpy(data.get() + coded_data_bit_length_start, device_bit_length, compression_blocks_count * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        cudaMemcpy(data.get() + host_out_start, device_compressed, compressed_data_size, cudaMemcpyDeviceToHost);

        compressed_size = compressed_data_size_final;

        cudaFree(device_uncompressed);
        cudaFree(device_compressed);
        cudaFree(device_bit_length);
        cudaFree(device_position_id);
        cudaFree(device_compressed_size);

        for (int i = 0; i < 10; i++)
        {
            printf("host %d\n", data.get()[host_out_start + i]);
        }

        return data;
    }

};
