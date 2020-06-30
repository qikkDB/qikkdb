#pragma once

#include <cstdio>
#include <chrono>
#include <stdexcept>
#include <stdint.h>
#include <inttypes.h>
#include <vector>
#include <cstdint>

class CompressionGPU
{

public:
    static const char CWARP_SIZE = 32;

    /// <summary>
    /// Compresses input data and fills output vector with compressed data
    /// </summary>
    /// <param name="hostUncompressed">Poiter to uncompressed data stored in host memory</param>
    /// <param name="uncompressedElementsCount">Number of elements of uncompressed data</param>
    /// <param name="hostCompressed">Compressed data vector in host memory</param>
    /// <param name="compressedElementsCount">Number of elements of compressed data</param>
    /// <param name="minValue">Minimum value of uncompressed data</param>
    /// <param name="maxValue">Maximum value of uncompressed data</param>
    /// <returns>Value representing result of compression</returns>
    template <typename T>
    static bool compressDataAAFL(T* const hostUncompressed,
                                 int64_t uncompressedElementsCount,
                                 std::vector<T>& hostCompressed,
                                 int64_t& compressedElementsCount,
                                 T minValue,
                                 T maxValue);

    /// <summary>
    /// Decompresses input data and fills output vector with decompressed data
    /// </summary>
    /// <param name="hostCompressed">Pointer to compressed data stored in host memory</param>
    /// <param name="compressedElementsCount">Number of elements of compressed data</param>
    /// <param name="hostUncompressed">Uncompressed data vector in host memory</param>
    /// <param name="uncompressedElementsCount">Number of elements of uncompressed data</param>
    /// <param name="minValue">Minimum value of uncompressed data</param>
    /// <param name="maxValue">Maximum value of uncompressed data</param>
    /// <returns>Value representing result of decompression</returns>
    template <typename T>
    static bool decompressDataAAFL(T* const hostCompressed,
                                   int64_t compressedElementsCount,
                                   std::vector<T>& hostUncompressed,
                                   int64_t& uncompressedElementsCount,
                                   T minValue,
                                   T maxValue);

    /// <summary>
    /// Decompresses input data directly on device and fills reserved space on device with decompressed data
    /// </summary>
    /// <param name="deviceCompressed">Pointer to compressed data stored in device memory</param>
    /// <param name="uncompressedElementsCount">Number of elements of uncompressed data</param>
    /// <param name="compressedElementsCount">Number of elements of compressed data</param>
    /// <param name="compressionBlocksCount">Number of elements of compression blocks</param>
    /// <param name="deviceUncompressed">Pointer to compressed data stored in device memory</param>
    /// <param name="minValue">Minimum value of uncompressed data</param>
    /// <param name="maxValue">Maximum value of uncompressed data</param>
    /// <returns>Value representing result of decompression</returns>
    template <typename T>
    static bool decompressDataAAFLOnDevice(T* const deviceCompressed,
                                           int64_t uncompressedElementsCount,
                                           int64_t compressedElementsCount,
                                           int64_t compressionBlocksCount,
                                           T* const deviceUncompressed,
                                           T minValue,
                                           T maxValue);
};
