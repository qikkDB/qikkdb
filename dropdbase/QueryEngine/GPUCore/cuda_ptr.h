#pragma once

#include "GPUMemory.cuh"

template<typename T>
class cuda_ptr
{
private:
	T * pointer_;

public:
	cuda_ptr(const cuda_ptr&) = delete;
	cuda_ptr& operator=(const cuda_ptr&) = delete;

	cuda_ptr(int32_t dataElementCount)
	{
		GPUMemory::alloc(&pointer_, dataElementCount);
	}

	cuda_ptr(int32_t dataElementCount, int value)
	{
		GPUMemory::allocAndSet(&pointer_, value, dataElementCount);
	}

	~cuda_ptr()
	{
		if (pointer_ != nullptr)
		{
			GPUMemory::free(pointer_);
		}
	}

	T * get() const
	{
		return pointer_;
	}
};
