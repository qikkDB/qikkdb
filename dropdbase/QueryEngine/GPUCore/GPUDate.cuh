#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "ErrorFlagSwapper.h"
#include "../Context.h"
#include "../QueryEngineError.h"
#include "MaybeDeref.cuh"

namespace DateOperations
{
	struct year
	{
		template<typename T>
		__device__ T operator()(T a) const
		{
			return a;
		}
	};

	struct month
	{
		template<typename T>
		__device__ T operator()(T a) const
		{
			return a;
		}
	};

	struct day
	{
		template<typename T>
		__device__ T operator()(T a) const
		{
			return a;
		}
	};

	struct hour
	{
		template<typename T>
		__device__ T operator()(T a) const
		{
			return a;
		}
	};

	struct minute
	{
		template<typename T>
		__device__ T operator()(T a) const
		{
			return a;
		}
	};

	struct second
	{
		template<typename T>
		__device__ T operator()(T a) const
		{
			return a;
		}
	};
}

class GPUDate
{

};