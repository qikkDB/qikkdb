#pragma once

#include <vector>
#include "../BlockBase.h"
#include "GPUCompression.cuh"

class Compression
{
public:
	
	template<class T>
	static std::unique_ptr<BlockBase<T>> CompressBlock(BlockBase<T>& block)
	{
		int64_t compressed_size;
		CompressionGPU comp;
		std::vector<T> compressed;
		
		if (comp.compressDataAAFL(block.GetData(), (int64_t)block.GetSize(), compressed, compressed_size))
		{
			return std::make_unique<BlockBase<T>>(compressed, block.GetColumn());
		}
		else
		{
			return std::make_unique<BlockBase<T>>(compressed, block.GetColumn());
		}
		//std::vector<T> v(block.GetData(), block.GetData() + block.GetSize());
		//return std::make_unique<BlockBase<T>>(v, block.GetColumn());
	}
};