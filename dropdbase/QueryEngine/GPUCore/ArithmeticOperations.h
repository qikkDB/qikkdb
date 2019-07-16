/// Namespace for arithmetic operation generic functors
namespace ArithmeticOperations
{
	/// Arithmetic operation add
	struct add
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			// if none of the input operands are float
			if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
			{
				// Check for overflow
				if (((b > V{ 0 }) && (a > (max - b))) ||
					((b < V{ 0 }) && (a < (min - b))))
				{
					atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
					return GetNullConstant<T>();
				}
			}
			return a + b;
		}
	};

	struct addNoCheck
	{
		static constexpr bool isMonotonous = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b) const
		{
			return a + b;
		}
	};

	/// Arithmetic operation subtraction
	struct sub
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			// if none of the input operands are float
			if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
			{
				// Check for overflow
				if (((b > V{ 0 }) && (a < (min + b))) ||
					((b < V{ 0 }) && (a > (max + b))))
				{
					atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
					return GetNullConstant<T>();
				}
			}
			return a - b;
		}
	};

	struct subNoCheck
	{
		static constexpr bool isMonotonous = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b) const
		{
			return a - b;
		}
	};

	/// Arithmetic operation multiply
	struct mul
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			// if none of the input operands are float
			if (!std::is_floating_point<U>::value && !std::is_floating_point<V>::value)
			{
				// Check for overflow
				if (a > U{ 0 })
				{
					if (b > V{ 0 })
					{
						if (a > (max / b))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
							return GetNullConstant<T>();
						}
					}
					else
					{
						if (b < (min / a))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
							return GetNullConstant<T>();
						}
					}
				}
				else
				{
					if (b > V{ 0 })
					{
						if (a < (min / b))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
							return GetNullConstant<T>();
						}
					}
					else
					{
						if ((a != U{ 0 }) && (b < (max / a)))
						{
							atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_INTEGER_OVERFLOW_ERROR));
							return GetNullConstant<T>();
						}
					}
				}
			}
			return a * b;
		}
	};

	struct mulNoCheck
	{
		static constexpr bool isMonotonous = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b) const
		{
			return a * b;
		}
	};

	/// Arithmetic operation divide
	struct div
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			if (b == V{ 0 })
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR));
				return GetNullConstant<T>();
			}
			else
			{
				return a / b;
			}
		}
	};

	struct divNoCheck
	{
		static constexpr bool isMonotonous = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b) const
		{
			return a / b;
		}
	};

	/// Arithmetic operation modulo
	struct mod
	{
		template<typename T, typename U, typename V>
		__device__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			//modulo is not defined for floating point type
			static_assert(!std::is_floating_point<U>::value && !std::is_floating_point<V>::value,
				"None of the input columns of operation modulo cannot be floating point type!");

			// Check for zero division
			if (b == V{ 0 })
			{
				atomicExch(errorFlag, static_cast<int32_t>(QueryEngineErrorType::GPU_DIVISION_BY_ZERO_ERROR));
				return GetNullConstant<T>();
			}

			return a % b;
		}
	};

	struct modNoCheck
	{
		static constexpr bool isMonotonous = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b) const
		{
			//modulo is not defined for floating point type
			static_assert(!std::is_floating_point<U>::value && !std::is_floating_point<V>::value,
				"None of the input columns of operation modulo cannot be floating point type!");

			return a % b;
		}
	};

	/// Bitwise operation and
	struct bitwiseAnd
	{
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
		{
			return a & b;
		}
	};

	struct bitwiseAndNoCheck
	{
		static constexpr bool isMonotonous = false;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b)
		{
			return a & b;
		}
	};

	/// Bitwise operation or
	struct bitwiseOr
	{
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
		{
			return a | b;
		}
	};

	struct bitwiseOrNoCheck
	{
		static constexpr bool isMonotonous = false;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b)
		{
			return a | b;
		}
	};

	/// Bitwise operation xor
	struct bitwiseXor
	{
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
		{
			return a ^ b;
		}
	};

	struct bitwiseXorNoCheck
	{
		static constexpr bool isMonotonous = false;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b)
		{
			return a ^ b;
		}
	};

	/// Bitwise operation left shift
	struct bitwiseLeftShift
	{
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
		{
			return a << b;
		}
	};

	struct bitwiseLeftShiftNoCheck
	{
		static constexpr bool isMonotonous = false;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b)
		{
			return a << b;
		}
	};

	/// Bitwise operation right shift
	struct bitwiseRightShift
	{
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max)
		{
			return a >> b;
		}
	};

	struct bitwiseRightShiftNoCheck
	{
		static constexpr bool isMonotonous = false;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b)
		{
			return a >> b;
		}
	};

	/// Mathematical function logarithm
	struct logarithm
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			return logf(a) / logf(b);
		}
	};

	struct logarithmNoCheck
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b) const
		{
			return logf(a) / logf(b);
		}
	};

	/// Mathematical function arcus tangent
	struct arctangent2
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			return atan2f(a, b);
		}
	};

	struct arctangent2NoCheck
	{
		static constexpr bool isMonotonous = true;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b) const
		{
			return atan2f(a, b);
		}
	};

	/// Mathematical function power
	struct power
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			return powf(a, b);
		}
	};

	struct powerNoCheck
	{
		static constexpr bool isMonotonous = false;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b) const
		{
			return powf(a, b);
		}
	};

	/// Mathematical function root
	struct root
	{
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b, int32_t* errorFlag, T min, T max) const
		{
			return powf(a, 1 / b);
		}
	};

	struct rootNoCheck
	{
		static constexpr bool isMonotonous = false;
		static constexpr bool isFloatRetType = true;
		template<typename T, typename U, typename V>
		__device__ __host__ T operator()(U a, V b) const
		{
			return powf(a, 1 / b);
		}
	};
}