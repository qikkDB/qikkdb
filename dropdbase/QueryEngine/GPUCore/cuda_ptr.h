#pragma once

#include <memory>

#include "GPUMemory.cuh"

template<typename T>
class cuda_ptr
{
private:
	std::unique_ptr<T, void(*)(void *)> pointer_;

public:
	cuda_ptr(const cuda_ptr&) = delete;
	cuda_ptr& operator=(const cuda_ptr&) = delete;

	explicit cuda_ptr(int32_t dataElementCount) : pointer_(nullptr, &GPUMemory::free) // TODO bind CudaMemAllocator for correct graphic card
	{
		T * rawPointer;
		GPUMemory::alloc(&rawPointer, dataElementCount);
		pointer_.reset(rawPointer);
	}

	cuda_ptr(int32_t dataElementCount, int value) : pointer_(nullptr, &GPUMemory::free)
	{
		T * rawPointer;
		GPUMemory::allocAndSet(&rawPointer, value, dataElementCount);
		pointer_.reset(rawPointer);
	}

	T * get() const
	{
		return pointer_.get();
	}
};
