#pragma once

/// Functors for parallel binary filtration operations
namespace FilterConditions
{
	/// A greater than operator > functor
	struct greater
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a > b;
		}

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			return false;	// TODO
		}
	};

	/// A greater than or equal operator >= functor
	struct greaterEqual
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a >= b;
		}

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			return false;	// TODO
		}
	};

	/// A less than operator < functor
	struct less
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a < b;
		}

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			return false;	// TODO
		}
	};

	/// A less than or equal operator <= functor
	struct lessEqual
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a <= b;
		}

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			return false;	// TODO
		}
	};

	/// An equality operator == functor
	struct equal
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a == b;
		}

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			if (aLength != bLength)
			{
				return false;
			}
			else
			{
				for (int32_t j = 0; j < aLength; j++)
				{
					if (a[j] != b[j])
					{
						return false;
					}
				}
				return true;
			}
		}
	};

	/// An unequality operator != functor
	struct notEqual
	{
		template<typename T, typename U>
		__device__ __host__ int8_t operator()(T a, U b) const
		{
			return a != b;
		}

		__device__ bool compareStrings(char * a, int32_t aLength, char * b, int32_t bLength)
		{
			if (aLength != bLength)
			{
				return true;
			}
			else
			{
				for (int32_t j = 0; j < aLength; j++)
				{
					if (a[j] != b[j])
					{
						return true;
					}
				}
				return false;
			}
		}
	};
}