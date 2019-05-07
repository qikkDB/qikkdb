#pragma once

template<int N>
class GPUStack 
{
private:
	char stackArray[N];
	int sp;

public:

	__device__ GPUStack() :
		sp(0)
	{

	}

	template<typename T>
	__device__ void push(T value)
	{
		int alignedSize = sizeof(T) % 4 == 0 ? sizeof(T) : sizeof(T) + 4 - (sizeof(T) % 4);
		if (sp + alignedSize < N)
		{
			*(reinterpret_cast<T*>(stackArray + sp)) = value;
			sp += alignedSize;
		}
	}

	template<typename T>
	__device__ T pop()
	{
		int alignedSize = sizeof(T) % 4 == 0 ? sizeof(T) : sizeof(T) + 4 - (sizeof(T) % 4);
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
		int alignedSize = sizeof(T) % 4 == 0 ? sizeof(T) : sizeof(T) + 4 - (sizeof(T) % 4);
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