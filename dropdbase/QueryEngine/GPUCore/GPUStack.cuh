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
		if (sp + sizeof(T) < N)
		{
			*(reinterpret_cast<T*>(stackArray + sp)) = value;
			sp += sizeof(T);
		}
	}

	template<typename T>
	__device__ T pop()
	{
		if (sp - sizeof(T) < 0)
		{
			return T{};
		}
		else
		{
			sp -= sizeof(T);
			return *(reinterpret_cast<T*>(stackArray + sp));
		}
	}

	template<typename T>
	__device__ T top()
	{
		if (sp - sizeof(T) < 0)
		{
			return T{};
		}
		else
		{
			return *(reinterpret_cast<T*>(stackArray + sp - sizeof(T)));
		}
	}
};