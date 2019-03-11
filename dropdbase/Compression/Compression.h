#pragma once

#include <vector>
#include "../BlockBase.h"
#include "GPUCompression.cuh"

class Compression
{
public:
	//template<class T>
	//static std::unique_ptr<T[]> compressDataAAFL(T* host_uncompressed, size_t size, size_t& compressed_size) {
	//	return CompressionGPU::compressDataAAFL(host_uncompressed, size, compressed_size);
	//}

	template<class T>
	static std::unique_ptr<BlockBase<T>> CompressBlock(BlockBase<T>& block)
	{
		//int a = CompressionGPU::CompressBlock(block);
		//std::vector<T> v(block.GetData(), block.GetData() + block.GetSize());
		//return std::make_unique<BlockBase<T>>(v, block.GetColumn());
		int64_t compressed_size;
		std::unique_ptr<T[]> compressed = CompressionGPU::compressDataAAFL(block.GetData(), (int64_t)block.GetSize(), compressed_size);
		std::vector<T> v(compressed.get(), compressed.get() + compressed_size);
		return std::make_unique<BlockBase<T>>(v, block.GetColumn());
		//std::vector<T> v(block.GetData(), block.GetData() + block.GetSize());
		//return std::make_unique<BlockBase<T>>(v, block.GetColumn());
	}
};