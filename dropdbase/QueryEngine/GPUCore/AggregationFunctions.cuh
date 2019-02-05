#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Generic agg function functors
namespace AggregationFunctions
{
	struct min
	{
		template<typename T>
		__device__ void operator()(T *a, T b) const
		{
			atomicMin(a, b);
		}

		// Specialized atomicMin for int64_t
		__device__ void operator()(int64_t *a, int64_t b) const
		{
			int64_t old = *a;
			int64_t expected;
			if (old <= b)
			{
				return;
			}

			do
			{
				expected = old;
				uint64_t ret = atomicCAS(reinterpret_cast<uint64_t*>(a), *reinterpret_cast<uint64_t*>(&expected), *reinterpret_cast<uint64_t*>(&b));
				old = *(int64_t*)&ret;
			} while (old != expected && old > b);
		}

		// Specialized atomicMin for double
		__device__ void operator()(double *a, double b) const
		{
			double old = *a;
			double expected;
			if (old <= b)
			{
				return;
			}

			do
			{
				expected = old;
				uint64_t ret = atomicCAS(reinterpret_cast<uint64_t*>(a), *reinterpret_cast<uint64_t*>(&expected), *reinterpret_cast<uint64_t*>(&b));
				old = *(double*)&ret;
			} while (old != expected && old > b);
		}

		// Specialized atomicMin for floats
		__device__ void operator()(float *a, float b) const
		{
			float old = *a;
			float expected;
			if (old <= b)
			{
				return;
			}

			do
			{
				expected = old;
				int32_t ret = atomicCAS(reinterpret_cast<int32_t*>(a), *reinterpret_cast<int32_t*>(&expected), *reinterpret_cast<int32_t*>(&b));
				old = *(float*)&ret;
			} while (old != expected && old > b);
		}

		template<typename T>
		static void agg(T *outValue, T *ACol, int32_t dataElementCount)
		{
			T *outValueGPUPointer = thrust::min_element(thrust::device(CudaMemAllocator::GetInstance()), ACol, ACol + dataElementCount);
			cudaDeviceSynchronize();

			// Copy the generated output to outValue (still in GPU)
			cudaMemcpy(outValue, outValueGPUPointer, sizeof(T), cudaMemcpyDeviceToDevice);
		}

		template<typename T>
		static constexpr T getInitValue()
		{
			return std::numeric_limits<T>::max();
		}
	};

	struct max
	{
		template<typename T>
		__device__ void operator()(T *a, T b) const
		{
			atomicMax(a, b);
		}

		// Specialized atomicMax for int64_t
		__device__ void operator()(int64_t *a, int64_t b) const
		{
			int64_t old = *a;
			int64_t expected;
			if (old >= b)
			{
				return;
			}

			do
			{
				expected = old;
				uint64_t ret = atomicCAS(reinterpret_cast<uint64_t*>(a), *reinterpret_cast<uint64_t*>(&expected), *reinterpret_cast<uint64_t*>(&b));
				old = *(int64_t*)&ret;
			} while (old != expected && old < b);
		}

		// Specialized atomicMax for double
		__device__ void operator()(double *a, double b) const
		{
			double old = *a;
			double expected;
			if (old >= b)
			{
				return;
			}

			do
			{
				expected = old;
				uint64_t ret = atomicCAS(reinterpret_cast<uint64_t*>(a), *reinterpret_cast<uint64_t*>(&expected), *reinterpret_cast<uint64_t*>(&b));
				old = *(double*)&ret;
			} while (old != expected && old < b);
		}

		// Specialized atomicMax for floats
		__device__ void operator()(float *a, float b) const
		{
			float old = *a;
			float expected;
			if (old >= b)
			{
				return;
			}

			do
			{
				expected = old;
				int32_t ret = atomicCAS(reinterpret_cast<int32_t*>(a), *reinterpret_cast<int32_t*>(&expected), *reinterpret_cast<int32_t*>(&b));
				old = *(float*)&ret;
			} while (old != expected && old < b);
		}

		template<typename T>
		static void agg(T *outValue, T *ACol, int32_t dataElementCount)
		{
			T *outValueGPUPointer = thrust::max_element(thrust::device(CudaMemAllocator::GetInstance()), ACol, ACol + dataElementCount);
			cudaDeviceSynchronize();

			// Copy the generated output to outValue (still in GPU)
			cudaMemcpy(outValue, outValueGPUPointer, sizeof(T), cudaMemcpyDeviceToDevice);
		}

		template<typename T>
		static constexpr T getInitValue()
		{
			return std::numeric_limits<T>::lowest();
		}
	};

	struct sum
	{
		template<typename T>
		__device__ void operator()(T *a, T b) const
		{
			atomicAdd(a, b);
		}

		__device__ void operator()(int64_t *a, int64_t b) const
		{
			atomicAdd(reinterpret_cast<uint64_t*>(a), *reinterpret_cast<uint64_t*>(&b));
		}

		// Specialized atomicAdd for double
		__device__ void operator()(double *a, double b) const
		{
			double old = *a;
			double expected;
			double newValue;
			if (old >= b)
			{
				return;
			}

			do
			{
				expected = old;
				newValue = *a + b;
				uint64_t ret = atomicCAS(reinterpret_cast<uint64_t*>(a), *reinterpret_cast<uint64_t*>(&expected), *reinterpret_cast<uint64_t*>(&newValue));
				old = *(double*)&ret;
			} while (old != expected && old < b);
		}

		template<typename T>
		static void agg(T *outValue, T *ACol, int32_t dataElementCount)
		{
			// Kernel calls here
			T outValueHost = thrust::reduce(thrust::device(CudaMemAllocator::GetInstance()), ACol, ACol + dataElementCount, T{ 0 }, thrust::plus<T>());
			cudaDeviceSynchronize();

			// Copy the generated output to outValue (still in GPU)
			GPUMemory::copyHostToDevice(outValue, &outValueHost, 1);
		}

		template<typename T>
		static constexpr T getInitValue()
		{
			return T{ 0 };
		}
	};

	struct avg
	{
		template<typename T>
		__device__ void operator()(T *a, T b) const
		{
			atomicAdd(a, b);
		}

		__device__ void operator()(int64_t *a, int64_t b) const
		{
			atomicAdd(reinterpret_cast<uint64_t*>(a), *reinterpret_cast<uint64_t*>(&b));
		}

		// Specialized atomicAdd for double
		__device__ void operator()(double *a, double b) const
		{
			double old = *a;
			double expected;
			double newValue;
			do
			{
				expected = old;
				newValue = expected + b;
				uint64_t ret = atomicCAS(reinterpret_cast<uint64_t*>(a), *reinterpret_cast<uint64_t*>(&expected), *reinterpret_cast<uint64_t*>(&newValue));
				old = *(double*)&ret;
			} while (old != expected);
		}

		template<typename T>
		static void agg(T *outValue, T *ACol, int32_t dataElementCount)
		{
			T outValueHost = thrust::reduce(thrust::device(CudaMemAllocator::GetInstance()), ACol, ACol + dataElementCount, (T)0, thrust::plus<T>());
			outValueHost /= dataElementCount;
			cudaDeviceSynchronize();

			// Copy the generated output to outValue (still in GPU)
			GPUMemory::copyHostToDevice(outValue, &outValueHost, 1);
		}

		template<typename T>
		static constexpr T getInitValue()
		{
			return T{ 0 };
		}
	};

	struct count
	{
		template<typename T>
		__device__ void operator()(T *a, T b) const
		{
			// empty
		}

		template<typename T>
		static void agg(T *outValue, T *ACol, int32_t dataElementCount)
		{
			// TODO, make this function more useful
			T temp = dataElementCount;
			GPUMemory::copyHostToDevice(outValue, &temp, 1);
		}

		template<typename T>
		static constexpr T getInitValue()
		{
			return T{ 0 };
		}
	};
}