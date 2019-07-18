#include <cmath>

/// Namespace for unary arithmetic operation generic functors
namespace ArithmeticUnaryOperations
{
	/// Arithmetic unary minus
	struct minus
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = false;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return -a;
		}
	};

	/// Arithmetic unary absolute
	struct absolute
	{
		static constexpr bool isMonotonous = false;
		static constexpr bool isFloatRetType = false;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return abs(a);
		}
	};

	/// Mathematical function sine
	struct sine
	{
		static constexpr bool isMonotonous = false;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return sinf(a);
		}
	};

	/// Mathematical function cosine
	struct cosine
	{
		static constexpr bool isMonotonous = false;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return cosf(a);
		}
	};

	/// Mathematical function tangent
	struct tangent
	{
		static constexpr bool isMonotonous = false;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return tanf(a);
		}
	};

	/// Mathematical function cotangent
	struct cotangent
	{
		static constexpr bool isMonotonous = false;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return 1.0f / tanf(a);
		}
	};

	/// Mathematical function arcus sine
	struct arcsine
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return asinf(a);
		}
	};

	/// Mathematical function arcus cosine
	struct arccosine
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return acosf(a);
		}
	};

	/// Mathematical function arcus tangent
	struct arctangent
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return atanf(a);
		}
	};

	/// Mathematical function logarithm with base 10
	struct logarithm10
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return log10f(a);
		}
	};

	/// Mathematical function logarithm with base e
	struct logarithmNatural
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return logf(a);
		}
	};

	/// Mathematical function exponential
	struct exponential
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return expf(a);
		}
	};

	/// Mathematical function square root
	struct squareRoot
	{
		static constexpr bool isMonotonous = false;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return sqrtf(a);
		}
	};

	/// Mathematical function square
	struct square
	{
		static constexpr bool isMonotonous = false;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U a) const
		{
			return powf(a, 2);
		}
	};

	/// Mathematical function sign
	struct sign
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = false;
		template<typename T, typename U>
		__device__ __host__ T operator()(U val) const
		{
			return (U{ 0 } < val) - (val < U{ 0 });
		}
	};

	/// Mathematical function round
	struct round
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U val) const
		{
			return roundf(val);
		}
	};

	/// Mathematical function floor
	struct floor
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U val) const
		{
			return floorf(val);
		}
	};

	/// Mathematical function ceil
	struct ceil
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U>
		__device__ __host__ T operator()(U val) const
		{
			return ceilf(val);
		}
	};
}