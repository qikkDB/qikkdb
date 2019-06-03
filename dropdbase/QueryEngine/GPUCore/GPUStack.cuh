#pragma once

template<int N>
class GPUStack 
{
private:
	char stackArray[N];
	int sp;
	const int ALIGN_BYTES = 8;
public:

	__device__ GPUStack() :
		sp(0)
	{

	}

	template<typename T>
	__device__ void push(T value)
	{
		int alignedSize = sizeof(T) % ALIGN_BYTES == 0 ? sizeof(T) : sizeof(T) + ALIGN_BYTES - (sizeof(T) % ALIGN_BYTES);
		if (sp + alignedSize < N)
		{
			*(reinterpret_cast<T*>(stackArray + sp)) = value;
			sp += alignedSize;
		}
	}

	template<typename T>
	__device__ T pop()
	{
		int alignedSize = sizeof(T) % ALIGN_BYTES == 0 ? sizeof(T) : sizeof(T) + ALIGN_BYTES - (sizeof(T) % ALIGN_BYTES);
		if (sp - alignedSize < 0)
		{
			return T{};
		}
		else
		{
			sp -= alignedSize;
			return *(reinterpret_cast<T*>(stackArray + sp));
		}
	}

	template<typename T>
	__device__ T top()
	{
		int alignedSize = sizeof(T) % ALIGN_BYTES == 0 ? sizeof(T) : sizeof(T) + ALIGN_BYTES - (sizeof(T) % ALIGN_BYTES);
		if (sp - alignedSize < 0)
		{
			return T{};
		}
		else
		{
			return *(reinterpret_cast<T*>(stackArray + sp - alignedSize));
		}
	}
};